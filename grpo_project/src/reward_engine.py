# src/reward_engine.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class MultiObjectiveRewardEngine:
    def __init__(self, device="cuda:2"):
        self.device = device
        self.model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        
        print(f"Loading Reward Model on {self.device}...")
        # Load in bfloat16 to save memory (approx 16GB VRAM)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        
        # Define Objective Mapping
        # ArmoRM outputs a score vector. We map indices to concepts.
        self.obj_map = {
            "helpfulness": self.model.attributes.index("ultrafeedback-helpfulness"),
            "truthfulness": self.model.attributes.index("ultrafeedback-truthfulness"),
            "safety": self.model.attributes.index("beavertails-is_safe"),
            "following": self.model.attributes.index("ultrafeedback-instruction_following")
        }
        self.num_objectives = len(self.obj_map)

    def score(self, prompts, responses):
        """
        Input: List[str] prompts, List[str] responses
        Output: Tensor [Total_Samples, 4]
        """
        # Prepare ArmoRM chat format
        pairs = []
        for p, r in zip(prompts, responses):
            pairs.append([
                {"role": "user", "content": p},
                {"role": "assistant", "content": r}
            ])
            
        inputs = self.tokenizer.apply_chat_template(
            pairs, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model(inputs)
            # Extract relevant columns
            indices = list(self.obj_map.values())
            # ArmoRM outputs can be raw logits; we use them directly as rewards
            rewards = output.rewards[:, indices]
            
        return rewards
