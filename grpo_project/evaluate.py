"""
Evaluation Script for Principled GRPO

Compares the trained model against the baseline (reference) model on held-out prompts.
Outputs:
  - Reward score comparison
  - Qualitative response examples
  - Win rate vs baseline
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import json
from datetime import datetime

# Configuration
EVAL_PROMPTS = 50  # Number of prompts to evaluate
MODEL_PATH = "principled_grpo_model"  # Path to trained model
DEVICE = "cuda:0"
REWARD_DEVICE = "cuda:2"

def load_models():
    """Load both trained and baseline models."""
    print("Loading trained model...")
    trained_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        device_map=DEVICE
    )
    trained_model.eval()
    
    print("Loading baseline model...")
    baseline_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        device_map=DEVICE
    )
    baseline_model.eval()
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return trained_model, baseline_model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    """Generate a response from a model."""
    formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def get_reward_scores(prompts, responses, reward_engine):
    """Get reward scores for prompt-response pairs."""
    return reward_engine.get_rewards(prompts, responses)


def main():
    # Import reward engine
    from src.reward_engine import IsolatedRewardEngine
    
    print("="*60)
    print("  GRPO Evaluation: Trained vs Baseline")
    print("="*60)
    
    # Load models
    trained_model, baseline_model, tokenizer = load_models()
    
    # Load reward engine
    print(f"\nLoading Reward Engine on {REWARD_DEVICE}...")
    reward_engine = IsolatedRewardEngine(device=REWARD_DEVICE)
    
    # Load evaluation dataset (use validation split if available, else sample from train)
    print("\nLoading evaluation prompts...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
    # Use prompts from the END of the dataset (not seen during training)
    eval_indices = range(len(dataset) - EVAL_PROMPTS, len(dataset))
    
    results = {
        "trained_rewards": [],
        "baseline_rewards": [],
        "trained_wins": 0,
        "baseline_wins": 0,
        "ties": 0,
        "examples": []
    }
    
    print(f"\nEvaluating on {EVAL_PROMPTS} held-out prompts...\n")
    
    for idx in tqdm(eval_indices, desc="Evaluating"):
        prompt = dataset[idx]['prompt']
        
        # Generate responses
        trained_response = generate_response(trained_model, tokenizer, prompt)
        baseline_response = generate_response(baseline_model, tokenizer, prompt)
        
        # Get rewards
        trained_reward = reward_engine.get_rewards([prompt], [trained_response]).item()
        baseline_reward = reward_engine.get_rewards([prompt], [baseline_response]).item()
        
        results["trained_rewards"].append(trained_reward)
        results["baseline_rewards"].append(baseline_reward)
        
        # Determine winner
        if trained_reward > baseline_reward + 0.01:
            results["trained_wins"] += 1
            winner = "TRAINED"
        elif baseline_reward > trained_reward + 0.01:
            results["baseline_wins"] += 1
            winner = "BASELINE"
        else:
            results["ties"] += 1
            winner = "TIE"
        
        # Store examples (first 5)
        if len(results["examples"]) < 5:
            results["examples"].append({
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "trained_response": trained_response[:300] + "..." if len(trained_response) > 300 else trained_response,
                "baseline_response": baseline_response[:300] + "..." if len(baseline_response) > 300 else baseline_response,
                "trained_reward": trained_reward,
                "baseline_reward": baseline_reward,
                "winner": winner
            })
    
    # Cleanup
    reward_engine.shutdown()
    
    # Calculate statistics
    avg_trained = sum(results["trained_rewards"]) / len(results["trained_rewards"])
    avg_baseline = sum(results["baseline_rewards"]) / len(results["baseline_rewards"])
    win_rate = results["trained_wins"] / EVAL_PROMPTS * 100
    
    # Print results
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"\n📊 Reward Scores:")
    print(f"   Trained Model:  {avg_trained:.4f} (avg)")
    print(f"   Baseline Model: {avg_baseline:.4f} (avg)")
    print(f"   Improvement:    {(avg_trained - avg_baseline):.4f} ({(avg_trained/avg_baseline - 1)*100:.1f}%)")
    
    print(f"\n🏆 Win Rate:")
    print(f"   Trained Wins:   {results['trained_wins']} ({win_rate:.1f}%)")
    print(f"   Baseline Wins:  {results['baseline_wins']} ({100 - win_rate - results['ties']/EVAL_PROMPTS*100:.1f}%)")
    print(f"   Ties:           {results['ties']} ({results['ties']/EVAL_PROMPTS*100:.1f}%)")
    
    print("\n" + "="*60)
    print("  EXAMPLE COMPARISONS")
    print("="*60)
    
    for i, ex in enumerate(results["examples"], 1):
        print(f"\n--- Example {i} [{ex['winner']}] ---")
        print(f"📝 Prompt: {ex['prompt']}")
        print(f"\n🤖 Trained (reward={ex['trained_reward']:.3f}):")
        print(f"   {ex['trained_response']}")
        print(f"\n📦 Baseline (reward={ex['baseline_reward']:.3f}):")
        print(f"   {ex['baseline_response']}")
    
    # Save results to JSON
    output_file = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "avg_trained_reward": avg_trained,
            "avg_baseline_reward": avg_baseline,
            "improvement": avg_trained - avg_baseline,
            "win_rate": win_rate,
            "trained_wins": results["trained_wins"],
            "baseline_wins": results["baseline_wins"],
            "ties": results["ties"],
            "examples": results["examples"]
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
