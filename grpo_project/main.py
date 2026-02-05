import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Custom Imports
from src.reward_engine import MultiObjectiveRewardEngine
from src.utils import get_batch_logps, sample_weights, compute_grouped_advantages
from src.loss import D3POGRPOLoss

# --- CONFIG ---
POLICY_DEVICE = "cuda:0"
REF_DEVICE = "cuda:1"
REWARD_DEVICE = "cuda:2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--beta_kl", type=float, default=0.05)
    args = parser.parse_args()

    # 1. Load Policy (GPU 0)
    print(f"Loading Policy on {POLICY_DEVICE}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map=POLICY_DEVICE 
    )
    # Important: Set padding side for generation safety
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing=True,
    )
    model.train()

    # 2. Load Reference (GPU 1)
    print(f"Loading Reference on {REF_DEVICE}...")
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map=REF_DEVICE
    )
    ref_model.eval()

    # 3. Reward Engine (GPU 2)
    reward_engine = MultiObjectiveRewardEngine(device=REWARD_DEVICE)

    # 4. Data & Optimizer
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    loss_fn = D3POGRPOLoss(clip_eps=0.2)
    
    step_count = 0
    pbar = tqdm(total=args.max_steps)

    while step_count < args.max_steps:
        # --- A. Data Prep ---
        idx = (step_count * args.batch_size) % len(dataset)
        batch_prompts = dataset[idx : idx + args.batch_size]['prompt']
        
        # Sample Weights
        weights = sample_weights(len(batch_prompts), device=POLICY_DEVICE) # [Batch, 4]
        
        # Format Prompts with System Instructions
        formatted_prompts = []
        for p, w in zip(batch_prompts, weights.tolist()):
            sys = f"Priorities: [Helpful: {w[0]:.2f}, Truth: {w[1]:.2f}, Safe: {w[2]:.2f}, Follow: {w[3]:.2f}]"
            formatted_prompts.append(
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        # Replicate for Group Size
        group_prompts = [p for p in formatted_prompts for _ in range(args.group_size)]
        
        # Tokenize (Left Pad for Generation)
        inputs = tokenizer(group_prompts, return_tensors="pt", padding=True, truncation=True).to(POLICY_DEVICE)

        # --- B. Generation (Rollout) ---
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decouple Prompt & Response for Scoring
        gen_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # --- C. Reward & Advantage ---
        # Score on GPU 2
        raw_rewards = reward_engine.score(group_prompts, gen_texts).to(POLICY_DEVICE) # [B*G, 4]
        
        # CRITICAL: Create Attention Mask for the FULL sequence (Prompt + Gen)
        # The 'outputs' tensor contains the full sequence. We need a mask for it.
        # 1 for valid token, 0 for pad.
        full_attention_mask = (outputs != tokenizer.pad_token_id).long().to(POLICY_DEVICE)

        # --- D. KL Calculation (GPU 1) ---
        # Move full outputs to Ref Device
        ref_inputs = outputs.to(REF_DEVICE)
        ref_mask = full_attention_mask.to(REF_DEVICE)
        
        with torch.no_grad():
            ref_out = ref_model(input_ids=ref_inputs, attention_mask=ref_mask)
            ref_logps = get_batch_logps(ref_out.logits, ref_inputs, ref_mask).to(POLICY_DEVICE)

        # --- E. Old Logprobs (GPU 0) ---
        with torch.no_grad():
            curr_out = model(input_ids=outputs, attention_mask=full_attention_mask)
            old_logps = get_batch_logps(curr_out.logits, outputs, full_attention_mask)

        # Compute KL Penalty & Net Reward
        kl_penalty = old_logps - ref_logps
        rewards_net = raw_rewards - (args.beta_kl * kl_penalty.unsqueeze(1))
        
        # Advantages
        adv = compute_grouped_advantages(rewards_net.view(-1, args.group_size, 4))
        adv = adv.view(-1, 4) # Flatten
        
        # Expand Weights to match Group size
        w_expanded = weights.unsqueeze(1).expand(-1, args.group_size, -1).reshape(-1, 4)

        # --- F. PPO Training Loop ---
        for _ in range(args.ppo_epochs):
            model_out = model(input_ids=outputs, attention_mask=full_attention_mask)
            new_logps = get_batch_logps(model_out.logits, outputs, full_attention_mask)
            
            loss = loss_fn(new_logps, old_logps, adv, w_expanded)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step_count += 1
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.4f}")

    model.save_pretrained("final_d3po_model")

if __name__ == "__main__":
    main()
