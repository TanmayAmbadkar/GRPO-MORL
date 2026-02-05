# src/loss.py
import torch

class D3POGRPOLoss:
    """
    Computes weighted PPO loss.
    L = Sum_over_objs( w_i * PPO_Loss(Adv_i) )
    """
    def __init__(self, clip_eps=0.2):
        self.clip_eps = clip_eps

    def __call__(self, logprobs, old_logprobs, advantages, weights):
        """
        logprobs: [Batch*Group] (Flattened)
        old_logprobs: [Batch*Group] (Flattened)
        advantages: [Batch*Group, Num_Objs] (Flattened)
        weights: [Batch*Group, Num_Objs] (Flattened, Repeated)
        """
        # 1. Ratio
        # exp(new - old)
        ratio = torch.exp(logprobs - old_logprobs)
        
        total_loss = 0
        num_objs = advantages.shape[1]
        
        # 2. Iterate Objectives
        for i in range(num_objs):
            adv_i = advantages[:, i]
            w_i = weights[:, i]
            
            # PPO Clipping
            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_i
            
            # Minimize -Loss (Maximize Reward)
            obj_loss = -torch.min(surr1, surr2)
            
            # Weighted Sum
            total_loss += (w_i * obj_loss).mean()
            
        return total_loss
