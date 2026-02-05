# main.py
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse

# Import custom modules
from src.reward_engine import MultiObjectiveRewardEngine
from src.utils import get_batch_logps, sample_weights, compute_grouped_advantages
from src.loss import D3POGRPOLoss

# --- HARDWARE CONFIG ---
# GPU 0: Policy Model (Trainable, 4-bit LoRA)
# GPU 1: Reference Model (Frozen, 4-bit)
# GPU 2: Reward Model (Frozen, bfloat16)
# GPU 3: Spare (Used for offloading if needed, otherwise empty)
POLICY_DEVICE = "cuda:0"
REF_DEVICE = "cuda:1"
REWARD_DEVICE = "cuda:2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="Prompts per batch (multiplied by Group Size)")
    parser.add_argument("--group_size", type=int, default=4, help="Generations per prompt")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Inner training loop iterations")
    parser.add_argument("--beta_kl", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--max_steps", type=int, default=500, help="Total training steps")
    args = parser.parse_args()

    # 1. Load Policy Model (GPU 0)
    print(f"Loading Policy Model on {POLICY_DEVICE}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map=POLICY_DEVICE 
    )

    # Enable LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing=True,
    )
    model.train()

    # 2. Load Reference Model (GPU 1)
    print(f"Loading Reference Model on {REF_DEVICE}...")
    # We load a second copy specifically for reference. 
    # Unsloth is cheap on memory so 4bit + 4bit fits easily on separate cards.
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map=REF_DEVICE
    )
    ref_model.eval()

    # 3. Load Reward Engine (GPU 2)
    reward_engine = MultiObjectiveRewardEngine(device=REWARD_DEVICE)

    # 4. Data
    # Using simple UltraFeedback prompts
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    dataset = dataset.shuffle(seed=42)
    
    # 5. Setup Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    loss_fn = D3POGRPOLoss(clip_eps=0.2)
    
    # Training Loop
    step_count = 0
    pbar = tqdm(total=args.max_steps)

    while step_count < args.max_steps:
        # Sample batch of prompts
        # Note: In production, use a proper DataLoader. Here we slice manually for clarity.
        idx_start = (step_count * args.batch_size) % len(dataset)
        batch_data = dataset[idx_start : idx_start + args.batch_size]
        prompts_text = batch_data['prompt']
        
        current_bs = len(prompts_text)
        
        # A. Sample Weights [Batch, 4]
        weights = sample_weights(current_bs, num_objectives=4, device=POLICY_DEVICE)
        
        # B. Inject System Prompt (Formatting)
        formatted_prompts = []
        weight_list = weights.tolist()
        for p_txt, w in zip(prompts_text, weight_list):
            sys_msg = f"Priorities: [Helpfulness: {w[0]:.2f}, Truthfulness: {w[1]:.2f}, Safety: {w[2]:.2f}, Following: {w[3]:.2f}]"
            # Llama-3 Chat Format
            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{p_txt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            formatted_prompts.append(full_prompt)
            
        # C. Replicate for Group Generation
        # [P1, P1, P1, P1, P2, P2, P2, P2...]
        group_prompts = [p for p in formatted_prompts for _ in range(args.group_size)]
        
        # D. Rollout (Generate)
        inputs = tokenizer(group_prompts, return_tensors="pt", padding=True, truncation=True).to(POLICY_DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated text only (remove prompt)
        # We need this for the Reward Model
        gen_text_only = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # E. Score (Rewards) on GPU 2
        raw_rewards = reward_engine.score(group_prompts, gen_text_only) # [B*G, 4]
        
        # Move rewards to Policy Device for calculation
        rewards_tensor = raw_rewards.to(POLICY_DEVICE)
        
        # F. Compute Reference Logprobs (GPU 1) for KL
        # Move inputs/outputs to Ref Device
        ref_inputs = inputs.to(REF_DEVICE)
        ref_outputs_ids = outputs.to(REF_DEVICE)
        
        with torch.no_grad():
            ref_out = ref_model(input_ids=ref_outputs_ids, attention_mask=ref_inputs.attention_mask)
            ref_logps = get_batch_logps(ref_out.logits, ref_outputs_ids, ref_inputs.attention_mask)
            # Move back to Policy Device
            ref_logps = ref_logps.to(POLICY_DEVICE)

        # G. Compute 'Old' Logprobs (Initial Policy State) - on Policy Device
        with torch.no_grad():
            curr_out = model(input_ids=outputs, attention_mask=inputs.attention_mask)
            old_logps = get_batch_logps(curr_out.logits, outputs, inputs.attention_mask)
            
        # H. Compute KL and Final Rewards
        # KL = old_logps - ref_logps (Approx)
        kl_penalty = old_logps - ref_logps
        
        # Subtract KL from rewards. 
        # Note: We subtract KL from *every* objective or just treat it as a regularization?
        # Standard GRPO/PPO: R_total = R_score - beta * KL
        # We apply it to the rewards tensor broadcasting [B*G, 1]
        rewards_tensor = rewards_tensor - (args.beta_kl * kl_penalty.unsqueeze(1))
        
        # I. Compute Advantages (Group Normalization)
        # Reshape to [Batch, Group, Objs]
        rewards_reshaped = rewards_tensor.view(current_bs, args.group_size, -1)
        advantages = compute_grouped_advantages(rewards_reshaped)
        # Flatten back for training [Batch*Group, Objs]
        advantages = advantages.view(-1, 4)
        
        # Replicate weights to match [Batch*Group, Objs]
        # weights: [Batch, 4] -> [Batch, Group, 4] -> [Batch*Group, 4]
        weights_expanded = weights.unsqueeze(1).expand(-1, args.group_size, -1).reshape(-1, 4)
        
        # J. PPO Inner Loop
        for _ in range(args.ppo_epochs):
            # Forward pass
            model_out = model(input_ids=outputs, attention_mask=inputs.attention_mask)
            new_logps = get_batch_logps(model_out.logits, outputs, inputs.attention_mask)
            
            loss = loss_fn(new_logps, old_logps, advantages, weights_expanded)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        step_count += 1
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        # Periodic Save
        if step_count % 100 == 0:
            model.save_pretrained(f"checkpoints/step_{step_count}")

    print("Training Complete. Saving final model...")
    model.save_pretrained("final_d3po_model")

if __name__ == "__main__":
    main()
