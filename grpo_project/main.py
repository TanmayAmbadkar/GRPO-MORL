import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import os
import sys
from torch.utils.tensorboard import SummaryWriter

# --- FIX: Patch GenerationMixin to remove num_logits_to_keep before validation ---
import transformers.generation.utils as gen_utils
_ORIG_VALIDATE = gen_utils.GenerationMixin._validate_model_kwargs
def _patched_validate(self, model_kwargs):
    model_kwargs.pop("num_logits_to_keep", None)
    return _ORIG_VALIDATE(self, model_kwargs)
gen_utils.GenerationMixin._validate_model_kwargs = _patched_validate

# Custom Imports
from src.utils import get_batch_logps, compute_grpo_advantages
from src.loss import DualSignalGRPOLoss
from src.reward_engine import IsolatedRewardEngine

# --- CONFIG ---
TARGET_KL = 0.05       # Principled KL Budget
BETA_LR = 1e-2         # Learning rate for the dual variable (beta)
GROUP_SIZE = 2         # Reduced from 4 to save memory (2 models on same GPU)
MAX_STEPS = 500
POLICY_DEVICE = "cuda:0"
REF_DEVICE = "cuda:0"  # Same GPU - Unsloth has cross-device bug
REWARD_DEVICE = "cuda:1"

# Memory & Speed optimizations
MAX_NEW_TOKENS = 64

def main():
    # 1. Load Policy (GPU 0) - Uses Unsloth optimizations
    print(f"Loading Policy on {POLICY_DEVICE}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=1024,  # Reduced for speed
        dtype=None,
        load_in_4bit=True,
        device_map=POLICY_DEVICE
    )
    
    # Important: Set padding side for generation safety
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, 
        r=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # 2. Load Reference (GPU 1) - Uses Unsloth optimizations
    print(f"Loading Reference on {REF_DEVICE}...")
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        device_map=REF_DEVICE
    )
    ref_model.eval()

    # 3. Load Reward (GPU 2) - Isolated subprocess
    print(f"Loading Reward Engine on {REWARD_DEVICE} (isolated process)...")
    reward_engine = IsolatedRewardEngine(device=REWARD_DEVICE)

    # 4. Lagrangian Beta Setup
    initial_log_beta = torch.log(torch.tensor(0.1, device=POLICY_DEVICE))
    log_beta = initial_log_beta.detach().clone().requires_grad_(True)
    beta_optimizer = torch.optim.Adam([log_beta], lr=BETA_LR)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    loss_fn = DualSignalGRPOLoss(eps=0.2)
    
    # --- LOGGING SETUP ---
    writer = SummaryWriter(log_dir="logs/principled_grpo")
    
    print("Loading Dataset...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    print(f"Starting Training Loop (max_new_tokens={MAX_NEW_TOKENS}, group_size={GROUP_SIZE})...")
    try:
        pbar = tqdm(range(MAX_STEPS), desc="Training", unit="step")
        for step in pbar:
            # Clear CUDA cache to prevent OOM from memory fragmentation
            torch.cuda.empty_cache()
            
            # A. Generation (Rollout)
            batch = dataset[step : step + 1]
            prompts = [batch['prompt'][0]] * GROUP_SIZE
            
            formatted_prompts = [
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                for p in prompts
            ]
            
            inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(POLICY_DEVICE)
            
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, 
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            # Clamp outputs to valid token ID range
            # Use minimum of both models' vocab sizes to be safe
            policy_vocab = model.get_input_embeddings().weight.shape[0]
            ref_vocab = ref_model.get_input_embeddings().weight.shape[0]
            safe_vocab_size = min(policy_vocab, ref_vocab, len(tokenizer))
            
            max_token = outputs.max().item()
            min_token = outputs.min().item()
            
            # Debug: Print if tokens are out of range
            if max_token >= safe_vocab_size or min_token < 0:
                tqdm.write(f"⚠️  Clamping tokens: [{min_token}, {max_token}] -> [0, {safe_vocab_size-1}]")
                tqdm.write(f"    Policy vocab: {policy_vocab}, Ref vocab: {ref_vocab}, Tokenizer: {len(tokenizer)}")
            
            outputs = outputs.clone().clamp(0, safe_vocab_size - 1)
            
            # B. Get Signals
            responses = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            rewards = reward_engine.get_rewards(prompts, responses).to(POLICY_DEVICE)
            
            # C. LogProb & KL Calculation
            full_mask = (outputs != tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                # Move to ref device
                outputs_ref = outputs.to(REF_DEVICE)
                mask_ref = full_mask.to(REF_DEVICE)
                ref_logits = ref_model(input_ids=outputs_ref, attention_mask=mask_ref).logits
                ref_logps = get_batch_logps(ref_logits, outputs_ref, mask_ref).to(POLICY_DEVICE)
                
                old_logits = model(input_ids=outputs, attention_mask=full_mask).logits
                old_logps = get_batch_logps(old_logits, outputs, full_mask)
            
            kl_raw = old_logps - ref_logps
            
            # D. Advantage Calculation
            adv_rew = compute_grpo_advantages(rewards).squeeze(0)
            adv_kl = compute_grpo_advantages(kl_raw).squeeze(0)

            # E. Update Lagrangian Beta
            current_kl_avg = kl_raw.mean().detach()
            beta_loss = -log_beta.exp() * (current_kl_avg - TARGET_KL)
            beta_optimizer.zero_grad()
            beta_loss.backward()
            beta_optimizer.step()
            
            curr_beta = log_beta.exp().detach()

            # F. Policy Update
            model.train()
            new_logits = model(input_ids=outputs, attention_mask=full_mask).logits
            new_logps = get_batch_logps(new_logits, outputs, full_mask)
            ratio = torch.exp(new_logps - old_logps.detach())
            
            loss = loss_fn(ratio, adv_rew, adv_kl, curr_beta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # G. Logging
            reward_mean = rewards.mean().item()
            kl_mean = current_kl_avg.item()
            beta_val = curr_beta.item()
            loss_val = loss.item()
            
            writer.add_scalar("Reward/mean", reward_mean, step)
            writer.add_scalar("KL/mean", kl_mean, step)
            writer.add_scalar("KL/beta", beta_val, step)
            writer.add_scalar("Loss/total", loss_val, step)
            
            # Update progress bar
            pbar.set_postfix({
                "R": f"{reward_mean:.2f}",
                "KL": f"{kl_mean:.3f}",
                "β": f"{beta_val:.3f}",
                "L": f"{loss_val:.3f}"
            })

            # Detailed logging every 10 steps
            if step % 10 == 0:
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"📊 Step {step}/{MAX_STEPS}")
                tqdm.write(f"{'='*60}")
                tqdm.write(f"  Reward:     mean={reward_mean:.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")
                tqdm.write(f"  KL Div:     mean={kl_mean:.4f}, target={TARGET_KL}")
                tqdm.write(f"  Beta:       {beta_val:.6f}")
                tqdm.write(f"  Loss:       {loss_val:.6f}")
                tqdm.write(f"  Ratio:      mean={ratio.mean():.4f}, min={ratio.min():.4f}, max={ratio.max():.4f}")
                tqdm.write(f"  Tokens:     seq_len={outputs.shape[1]}, range=[{min_token}, {max_token}]")
                resp_preview = responses[0][:80] + "..." if len(responses[0]) > 80 else responses[0]
                tqdm.write(f"  Response:   \"{resp_preview}\"")
    finally:
        reward_engine.shutdown()
        writer.close()
        model.save_pretrained("principled_grpo_model")
        tokenizer.save_pretrained("principled_grpo_model")

if __name__ == "__main__":
    main()
