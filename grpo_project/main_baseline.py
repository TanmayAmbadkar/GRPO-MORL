import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import os
import sys
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transformers import LogitsProcessor, LogitsProcessorList

# --- FIX: Patch GenerationMixin to remove num_logits_to_keep before validation ---
import transformers.generation.utils as gen_utils
_ORIG_VALIDATE = gen_utils.GenerationMixin._validate_model_kwargs
def _patched_validate(self, model_kwargs):
    model_kwargs.pop("num_logits_to_keep", None)
    return _ORIG_VALIDATE(self, model_kwargs)
gen_utils.GenerationMixin._validate_model_kwargs = _patched_validate


class VocabClampLogitsProcessor(LogitsProcessor):
    """Masks logits beyond the safe vocabulary size to -inf during generation."""
    def __init__(self, safe_vocab_size: int):
        self.safe_vocab_size = safe_vocab_size
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[-1] > self.safe_vocab_size:
            scores[:, self.safe_vocab_size:] = float('-inf')
        return scores


# Custom Imports
from src.utils import get_batch_logps, compute_grpo_advantages
from src.reward_engine import IsolatedRewardEngine

# --- CONFIG ---
# BASELINE: Fixed beta instead of adaptive Lagrangian
BETA_KL = 0.1          # Fixed KL penalty coefficient
GROUP_SIZE = 2         # Reduced from 4 to save memory (2 models on same GPU)
MAX_STEPS = 500
CLIP_EPS = 0.2         # PPO clip epsilon
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

    # BASELINE: No Lagrangian beta - just fixed optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    
    # --- RUN DIRECTORY SETUP ---
    run_name = datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # TensorBoard logs in run directory
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
    
    # Setup file logging for stdout
    log_file = os.path.join(run_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, console_handler]
    )
    logger = logging.getLogger(__name__)
    
    # Save config to run directory
    config = {
        "BETA_KL": BETA_KL,
        "GROUP_SIZE": GROUP_SIZE,
        "MAX_STEPS": MAX_STEPS,
        "CLIP_EPS": CLIP_EPS,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "POLICY_DEVICE": POLICY_DEVICE,
        "REF_DEVICE": REF_DEVICE,
        "REWARD_DEVICE": REWARD_DEVICE,
    }
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config: {config}")
    
    # Model save path
    model_save_path = os.path.join(run_dir, "model")
    
    print("Loading Dataset...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    print(f"Starting BASELINE Training Loop (max_new_tokens={MAX_NEW_TOKENS}, group_size={GROUP_SIZE})...")
    try:
        pbar = tqdm(range(MAX_STEPS), desc="Baseline Training", unit="step")
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
            
            # Compute safe vocabulary size before generation
            policy_vocab = model.get_input_embeddings().weight.shape[0]
            ref_vocab = ref_model.get_input_embeddings().weight.shape[0]
            safe_vocab_size = min(policy_vocab, ref_vocab, len(tokenizer))
            
            # Create logits processor to prevent out-of-bounds tokens
            vocab_clamper = VocabClampLogitsProcessor(safe_vocab_size)
            
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, 
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([vocab_clamper]),
                )
            
            # Safety check: verify outputs are in valid range
            max_token = outputs.max().item()
            min_token = outputs.min().item()
            
            if max_token >= safe_vocab_size or min_token < 0:
                tqdm.write(f"⚠️  Unexpected out-of-range tokens: [{min_token}, {max_token}]")
                outputs = outputs.clone().clamp(0, safe_vocab_size - 1)
            
            # B. Get Signals
            responses = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            rewards = reward_engine.get_rewards(prompts, responses).to(POLICY_DEVICE)
            
            # C. LogProb & KL Calculation
            full_mask = (outputs != tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                # Transfer to reference device (same GPU in this config)
                outputs_ref = outputs.to(REF_DEVICE)
                mask_ref = full_mask.to(REF_DEVICE)
                
                ref_logits = ref_model(input_ids=outputs_ref, attention_mask=mask_ref).logits
                ref_logps = get_batch_logps(ref_logits, outputs_ref, mask_ref).to(POLICY_DEVICE)
                
                old_logits = model(input_ids=outputs, attention_mask=full_mask).logits
                old_logps = get_batch_logps(old_logits, outputs, full_mask)
            
            kl_raw = old_logps - ref_logps
            
            # D. BASELINE: Standard combined advantage (Reward - beta * KL)
            total_signal = rewards - BETA_KL * kl_raw
            advantages = compute_grpo_advantages(total_signal).squeeze(0)

            # E. BASELINE: Standard PPO Clip Loss (no Lagrangian update)
            model.train()
            new_logits = model(input_ids=outputs, attention_mask=full_mask).logits
            new_logps = get_batch_logps(new_logits, outputs, full_mask)
            ratio = torch.exp(new_logps - old_logps.detach())
            
            # PPO Clipped Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            loss = -torch.min(surr1, surr2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # F. Logging
            reward_mean = rewards.mean().item()
            kl_mean = kl_raw.mean().item()
            loss_val = loss.item()
            
            writer.add_scalar("Reward/mean", reward_mean, step)
            writer.add_scalar("KL/mean", kl_mean, step)
            writer.add_scalar("Loss/total", loss_val, step)
            
            # Update progress bar
            pbar.set_postfix({
                "R": f"{reward_mean:.2f}",
                "KL": f"{kl_mean:.3f}",
                "L": f"{loss_val:.3f}"
            })

            # Detailed logging every 10 steps
            if step % 10 == 0:
                log_msg = f"""
{'='*60}
📊 BASELINE Step {step}/{MAX_STEPS}
{'='*60}
  Reward:     mean={reward_mean:.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}
  KL Div:     mean={kl_mean:.4f}, beta={BETA_KL} (fixed)
  Loss:       {loss_val:.6f}
  Ratio:      mean={ratio.mean():.4f}, min={ratio.min():.4f}, max={ratio.max():.4f}
  Tokens:     seq_len={outputs.shape[1]}, range=[{min_token}, {max_token}]"""
                resp_preview = responses[0][:80] + "..." if len(responses[0]) > 80 else responses[0]
                log_msg += f'\n  Response:   "{resp_preview}"'
                
                # Write to both console and log file
                tqdm.write(log_msg)
                logger.info(log_msg)
    finally:
        reward_engine.shutdown()
        writer.close()
        
        # Save model to run directory
        logger.info(f"Saving model to {model_save_path}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logger.info("Training complete!")

if __name__ == "__main__":
    main()
