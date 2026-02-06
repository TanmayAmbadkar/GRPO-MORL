from unsloth import FastLanguageModel
import torch
import os
import sys
import importlib
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# --- FIX: Recover original Llama methods for the Reward Model ---
import transformers.models.llama.modeling_llama as modeling_llama
importlib.reload(modeling_llama)
_ORIG_MODEL_FORWARD = modeling_llama.LlamaModel.forward
_ORIG_LAYER_FORWARD = modeling_llama.LlamaDecoderLayer.forward
_ORIG_ATTN_FORWARD = modeling_llama.LlamaAttention.forward
_ORIG_RMS_FORWARD = modeling_llama.LlamaRMSNorm.forward
_ORIG_ROTARY_FORWARD = modeling_llama.LlamaRotaryEmbedding.forward

def fix_unsloth_generation(model):
    """
    Patches the model instance to ignore 'num_logits_to_keep' during generation.
    """
    if hasattr(model, "_original_forward_for_fix"):
        return model

    model._original_forward_for_fix = model.forward

    def _fixed_forward(*args, **kwargs):
        kwargs.pop("num_logits_to_keep", None)
        return model._original_forward_for_fix(*args, **kwargs)

    model.forward = _fixed_forward
    return model

def unpatch_model(model):
    """
    Restores original Hugging Face forward methods to a model instance
    to prevent conflicts with Unsloth's global monkey-patching.
    """
    from transformers.models.llama.modeling_llama import (
        LlamaModel, LlamaDecoderLayer, LlamaAttention, 
        LlamaRMSNorm, LlamaRotaryEmbedding
    )
    for name, module in model.named_modules():
        if isinstance(module, LlamaModel):
            module.forward = _ORIG_MODEL_FORWARD.__get__(module, LlamaModel)
        if isinstance(module, LlamaDecoderLayer):
            module.forward = _ORIG_LAYER_FORWARD.__get__(module, LlamaDecoderLayer)
        if isinstance(module, LlamaAttention):
            module.forward = _ORIG_ATTN_FORWARD.__get__(module, LlamaAttention)
        if isinstance(module, LlamaRMSNorm):
            module.forward = _ORIG_RMS_FORWARD.__get__(module, LlamaRMSNorm)
        if isinstance(module, LlamaRotaryEmbedding):
            module.forward = _ORIG_ROTARY_FORWARD.__get__(module, LlamaRotaryEmbedding)

# Custom Imports
from src.utils import get_batch_logps, compute_grpo_advantages
from src.reward_engine import RewardEngine

# --- CONFIG ---
BETA_KL = 0.1          # Fixed KL Penalty for Baseline
GROUP_SIZE = 4         # GRPO Group size
MAX_STEPS = 500
POLICY_DEVICE = "cuda:0"
REF_DEVICE = "cuda:1"
REWARD_DEVICE = "cuda:2"
CLIP_EPS = 0.2

def main():
    # 1. Load Policy (GPU 0)
    print(f"Loading Policy on {POLICY_DEVICE}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map=POLICY_DEVICE
    )
    fix_unsloth_generation(model)
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing=True,
    )
    
    # 2. Load Reference (GPU 1)
    print(f"Loading Reference on {REF_DEVICE}...")
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map=REF_DEVICE
    )
    fix_unsloth_generation(ref_model)
    ref_model.eval()

    # 3. Load Reward (GPU 2)
    print(f"Loading Reward Engine on {REWARD_DEVICE}...")
    reward_engine = RewardEngine(device=REWARD_DEVICE)
    # Apply the unpatch fix to the Reward Model
    unpatch_model(reward_engine.model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    
    # --- LOGGING SETUP ---
    writer = SummaryWriter(log_dir="logs/baseline_grpo")
    
    print("Loading Dataset...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    print("Starting Baseline Training Loop...")
    for step in range(MAX_STEPS):
        # A. Generation (Rollout)
        batch = dataset[step : step + 1]
        prompts = [batch['prompt'][0]] * GROUP_SIZE
        formatted_prompts = [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            for p in prompts
        ]
        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(POLICY_DEVICE)
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.8, pad_token_id=tokenizer.pad_token_id)
        
        # B. Get Signals
        responses = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        rewards = reward_engine.get_rewards(prompts, responses).to(POLICY_DEVICE)
        
        # C. LogProb & KL Calculation
        full_mask = (outputs != tokenizer.pad_token_id).long()
        with torch.no_grad():
            ref_logits = ref_model(input_ids=outputs.to(REF_DEVICE), attention_mask=full_mask.to(REF_DEVICE)).logits
            ref_logps = get_batch_logps(ref_logits, outputs.to(REF_DEVICE), full_mask.to(REF_DEVICE)).to(POLICY_DEVICE)
            old_logits = model(input_ids=outputs, attention_mask=full_mask).logits
            old_logps = get_batch_logps(old_logits, outputs, full_mask)
        
        kl_raw = old_logps - ref_logps
        
        # D. Standard Advantage Calculation (Combined)
        # Reverting to original GRPO: Adv(Reward - beta * KL)
        total_rewards = rewards - BETA_KL * kl_raw
        adv_total = compute_grpo_advantages(total_rewards).squeeze(0)

        # F. Policy Update
        model.train()
        new_logits = model(input_ids=outputs, attention_mask=full_mask).logits
        new_logps = get_batch_logps(new_logits, outputs, full_mask)
        ratio = torch.exp(new_logps - old_logps.detach())
        
        # Standard PPO Clip Loss
        surr1 = ratio * adv_total
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_total
        loss = -torch.min(surr1, surr2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # G. Logging
        writer.add_scalar("Reward/mean", rewards.mean().item(), step)
        writer.add_scalar("KL/mean", kl_raw.mean().item(), step)
        writer.add_scalar("Loss/total", loss.item(), step)

        if step % 10 == 0:
            print(f"Step {step} | Reward: {rewards.mean():.2f} | KL: {kl_raw.mean():.4f} | Loss: {loss.item():.4f}")

    writer.close()
    model.save_pretrained("baseline_grpo_model")
    tokenizer.save_pretrained("baseline_grpo_model")

if __name__ == "__main__":
    main()
