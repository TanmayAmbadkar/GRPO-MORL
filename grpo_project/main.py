import os
import sys
import logging
from datetime import datetime
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import LogitsProcessor, LogitsProcessorList, AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

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
from src.loss import DualSignalGRPOLoss
# note: IsolatedRewardEngine is removed; we load directly on the device now

# --- PRO CONFIG (4x A100-80GB) ---
TARGET_KL = 0.1        # Slightly looser constraint for larger models/batches
BETA_LR = 5e-2         # Faster reaction time for the controller
BETA_MAX = 0.693       # Clamp Beta at ~2.0 (exp(0.693)) to prevent model collapse
GROUP_SIZE = 16        # Massive group size (16 per GPU * 4 GPUs = 64 effective)
MAX_STEPS = 500
MAX_NEW_TOKENS = 2048  # Full Chain-of-Thought capacity
MAX_SEQ_LENGTH = 4096  # Context Window

def main():
    # 1. Initialize Accelerator (Handles DDP and Device Placement)
    accelerator = Accelerator(gradient_accumulation_steps=1)
    set_seed(42 + accelerator.process_index)
    device = accelerator.device

    # *** CRITICAL: Load Reward Model BEFORE importing Unsloth ***
    # Unsloth monkey-patches transformers LlamaAttention globally.
    # If we load the RM after Unsloth import, it breaks with "apply_qkv" error.
    if accelerator.is_main_process: print(f"Loading Reward Model on {device} (BEFORE Unsloth)...")
    rm_tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", use_fast=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"  # Use eager attention to avoid Unsloth patches
    ).eval()

    # NOW import Unsloth (this patches transformers globally)
    from unsloth import FastLanguageModel

    # Logging Setup (Only on Main Process)
    if accelerator.is_main_process:
        run_name = datetime.now().strftime("pro_run_%Y%m%d_%H%M%S")
        run_dir = os.path.join("logs", run_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
        
        # Save config
        config = {
            "TARGET_KL": TARGET_KL, "BETA_LR": BETA_LR, "GROUP_SIZE": GROUP_SIZE,
            "MAX_STEPS": MAX_STEPS, "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
            "LR": 1e-6, "TEMP": 1.2
        }
        with open(os.path.join(run_dir, "config.txt"), "w") as f:
            for k, v in config.items(): f.write(f"{k}: {v}\n")
    else:
        run_dir = None
        writer = None

    # 2. Load Policy (Full 4-bit LoRA on local rank)
    if accelerator.is_main_process: print(f"Loading Policy on {device}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,  # Native A100 format
        load_in_4bit=True,
        device_map={"": device}, # Force Unsloth to use local accelerator device
        attn_implementation="flash_attention_2"
    )
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, 
        r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # 3. Load Reference (Frozen 4-bit on local rank)
    if accelerator.is_main_process: print(f"Loading Reference on {device}...")
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={"": device},
        attn_implementation="flash_attention_2"
    )
    ref_model.eval()

    # 4. Reward Model already loaded above (before Unsloth import)

    # 5. Optimizer & Controller
    # Use very low LR for large group sizes
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    loss_fn = DualSignalGRPOLoss(eps=0.2)
    
    initial_log_beta = torch.log(torch.tensor(0.1, device=device))
    log_beta = initial_log_beta.detach().clone().requires_grad_(True)
    beta_optimizer = torch.optim.Adam([log_beta], lr=BETA_LR)

    # Prepare with Accelerator
    # Note: Unsloth models are technically PeftModels, accelerate handles them.
    model, optimizer = accelerator.prepare(model, optimizer)
    # Ref and Reward are frozen/eval, no need to prepare them for gradients

    # 6. Dataset
    if accelerator.is_main_process: print("Loading Dataset...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    # Training Loop
    if accelerator.is_main_process: print(f"Starting Pro Training Loop (G={GROUP_SIZE})...")
    
    # Simple sampler for DDP: split dataset by process index
    # (For a real run, use a DistributedSampler, but this is fine for linear iteration)
    local_indices = range(accelerator.process_index, len(dataset), accelerator.num_processes)
    
    step = 0
    pbar = tqdm(total=MAX_STEPS, disable=not accelerator.is_main_process)
    
    for idx in local_indices:
        if step >= MAX_STEPS: break
        
        # Clear Cache
        torch.cuda.empty_cache()
        
        # A. Generation
        prompt_text = dataset[idx]['prompt']
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Repeat prompt for Group Generation
        prompts = [formatted_prompt] * GROUP_SIZE
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(device)

        # Vocab Clamp Setup - use unwrap_model for DDP compatibility
        unwrapped_model = accelerator.unwrap_model(model)
        policy_vocab = unwrapped_model.get_input_embeddings().weight.shape[0]
        ref_vocab = ref_model.get_input_embeddings().weight.shape[0]
        safe_vocab_size = min(policy_vocab, ref_vocab, len(tokenizer))
        vocab_clamper = VocabClampLogitsProcessor(safe_vocab_size)

        model.eval()
        with torch.no_grad():
            # Use unwrapped model for generation (DDP doesn't expose .generate())
            outputs = unwrapped_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.2, # High Temp for G=16 exploration
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([vocab_clamper]),
            )

        # B. Get Signals (Reward)
        # Decode responses
        responses = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Score with Reward Model
        rm_pairs = [[{"role": "user", "content": prompt_text}, {"role": "assistant", "content": r}] for r in responses]
        rm_input_ids = rm_tokenizer.apply_chat_template(rm_pairs, return_tensors="pt", padding=True, truncation=True).to(device)
        # apply_chat_template returns a raw tensor, create attention mask manually
        rm_attention_mask = (rm_input_ids != rm_tokenizer.pad_token_id).long()
        with torch.no_grad():
            rewards = reward_model(input_ids=rm_input_ids, attention_mask=rm_attention_mask).rewards[:, 0]

        # C. LogProb & KL
        full_mask = (outputs != tokenizer.pad_token_id).long()
        with torch.no_grad():
            ref_logits = ref_model(input_ids=outputs, attention_mask=full_mask).logits
            ref_logps = get_batch_logps(ref_logits, outputs, full_mask)
            
            # Forward pass on Policy to get old_logps
            # Use accelerator.unwrap_model to call forward safely if needed
            old_logits = model(input_ids=outputs, attention_mask=full_mask).logits
            old_logps = get_batch_logps(old_logits, outputs, full_mask)

        kl_raw = old_logps - ref_logps # [G]

        # D. Advantage Calculation (Local to this GPU's Group)
        # In DDP, we can either normalize locally (Standard GRPO) or globally.
        # DeepSeek uses Local normalization (per prompt). Since each GPU has 1 prompt, we use local stats.
        adv_rew = compute_grpo_advantages(rewards.unsqueeze(0)).squeeze(0)
        adv_kl = compute_grpo_advantages(kl_raw.unsqueeze(0)).squeeze(0)

        # E. Update Lagrangian Beta
        current_kl_avg = kl_raw.mean().detach()
        beta_loss = -log_beta.exp() * (current_kl_avg - TARGET_KL)
        beta_optimizer.zero_grad()
        beta_loss.backward()
        beta_optimizer.step()
        
        # Clamp Beta to prevent explosion (Fix for Boom/Bust)
        with torch.no_grad():
            log_beta.clamp_(max=BETA_MAX) # Max Beta ~ 2.0
            
        curr_beta = log_beta.exp().detach()

        # F. Policy Update
        model.train()
        new_logits = model(input_ids=outputs, attention_mask=full_mask).logits
        new_logps = get_batch_logps(new_logits, outputs, full_mask)
        ratio = torch.exp(new_logps - old_logps.detach())
        
        loss = loss_fn(ratio, adv_rew, adv_kl, curr_beta)
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        # G. Logging (Main Process Only)
        if accelerator.is_main_process:
            step += 1
            pbar.update(1)
            
            reward_mean = rewards.mean().item()
            kl_val = current_kl_avg.item()
            beta_val = curr_beta.item()
            loss_val = loss.item()

            writer.add_scalar("Reward/mean", reward_mean, step)
            writer.add_scalar("KL/mean", kl_val, step)
            writer.add_scalar("KL/beta", beta_val, step)
            writer.add_scalar("Loss/total", loss_val, step)

            pbar.set_postfix({"R": f"{reward_mean:.2f}", "KL": f"{kl_val:.3f}", "B": f"{beta_val:.2f}"})

            if step % 10 == 0:
                tqdm.write(f"\nStep {step} | R: {reward_mean:.4f} | KL: {kl_val:.4f} | Beta: {beta_val:.4f} | L: {loss_val:.4f}")

    # Save
    if accelerator.is_main_process:
        print(f"Saving model to {os.path.join(run_dir, 'model')}")
        model_save = accelerator.unwrap_model(model)
        model_save.save_pretrained(os.path.join(run_dir, "model"))
        tokenizer.save_pretrained(os.path.join(run_dir, "model"))
        writer.close()

if __name__ == "__main__":
    main()