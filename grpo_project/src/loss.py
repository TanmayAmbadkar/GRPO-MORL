import torch

class D3POGRPOLoss:
    """
    Multi-Objective GRPO Loss.
    L = Sum( w_i * Clip(Adv_i) )
    """
    def __init__(self, clip_eps=0.2):
        self.clip_eps = clip_eps

    def __call__(self, logprobs, old_logprobs, advantages, weights):
        # 1. Ratio = exp(new - old)
        ratio = torch.exp(logprobs - old_logprobs)
        
        total_loss = 0
        num_objs = advantages.shape[1]
        
        # 2. Iterate over objectives
        for i in range(num_objs):
            adv_i = advantages[:, i]
            w_i = weights[:, i]
            
            # Standard PPO Clipping
            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_i
            
            # Objective Loss
            obj_loss = -torch.min(surr1, surr2)
            
            # Weighted Sum
            total_loss += (w_i * obj_loss).mean()
            
        return total_loss
