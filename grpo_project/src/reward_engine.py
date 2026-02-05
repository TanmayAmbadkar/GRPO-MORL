import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class MultiObjectiveRewardEngine:
    def __init__(self, device="cuda:2"):
        self.device = device
        self.model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        
        print(f"Loading Reward Model on {self.device}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        
        # Map ArmoRM attributes to our 4 objectives
        self.obj_map = {
            "helpfulness": self.model.attributes.index("ultrafeedback-helpfulness"),
            "truthfulness": self.model.attributes.index("ultrafeedback-truthfulness"),
            "safety": self.model.attributes.index("beavertails-is_safe"),
            "following": self.model.attributes.index("ultrafeedback-instruction_following")
        }

    def score(self, prompts, responses):
        """
        Returns: [Total_Samples, 4]
        """
        pairs = [[{"role": "user", "content": p}, {"role": "assistant", "content": r}] 
                 for p, r in zip(prompts, responses)]
            
        inputs = self.tokenizer.apply_chat_template(
            pairs, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model(inputs)
            indices = list(self.obj_map.values())
            rewards = output.rewards[:, indices]
            
        return rewards
