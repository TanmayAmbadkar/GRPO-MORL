import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.multiprocessing as mp

def _reward_worker(device, model_id, input_queue, output_queue):
    """
    Worker process that loads the Reward Model in complete isolation.
    """
    # This runs in a fresh process - no Unsloth patches here
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"[RewardWorker] Model loaded on {device}, ready for requests.")
    
    while True:
        item = input_queue.get()
        if item is None:  # Shutdown signal
            break
        prompts, responses = item
        
        pairs = [[{"role": "user", "content": p}, {"role": "assistant", "content": r}] 
                 for p, r in zip(prompts, responses)]
        
        inputs = tokenizer.apply_chat_template(
            pairs, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        # apply_chat_template returns a raw tensor (input_ids), create attention mask
        attention_mask = (inputs != tokenizer.pad_token_id).long()
        
        with torch.inference_mode():
            output = model(input_ids=inputs, attention_mask=attention_mask)
            rewards = output.rewards[:, 0].cpu()
        
        output_queue.put(rewards)
    
    print("[RewardWorker] Shutting down.")


class IsolatedRewardEngine:
    """
    Reward Engine that runs in a separate process to avoid Unsloth conflicts.
    """
    def __init__(self, device="cuda:2"):
        self.device = device
        self.model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        
        # Use spawn to get a fresh Python interpreter
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
            
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        
        print(f"Spawning Reward Model worker on {self.device}...")
        self.worker = mp.Process(
            target=_reward_worker,
            args=(self.device, self.model_id, self.input_queue, self.output_queue)
        )
        self.worker.start()
        
        # Wait for worker to be ready (first message will confirm)
        import time
        time.sleep(10)  # Give it time to load
    
    def get_rewards(self, prompts, responses):
        """Send request to worker and get rewards back."""
        self.input_queue.put((prompts, responses))
        rewards = self.output_queue.get()  # Blocking wait
        return rewards
    
    def shutdown(self):
        """Gracefully shutdown the worker process."""
        self.input_queue.put(None)
        self.worker.join(timeout=5)
        if self.worker.is_alive():
            self.worker.terminate()


# Alias for backwards compatibility
RewardEngine = IsolatedRewardEngine
