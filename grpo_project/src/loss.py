import torch

class DualSignalGRPOLoss:
    def __init__(self, eps=0.2):
        self.eps = eps

    def __call__(self, ratio, adv_rew, adv_kl, beta_kl):
        """
        ratio: pi_new / pi_old
        adv_rew: Advantage from Reward Model (Group-Normalized)
        adv_kl: Advantage from KL Divergence (Group-Normalized)
        beta_kl: Current Lagrangian multiplier
        """
        # Term 1: Maximize Reward (Standard PPO Clip)
        surr1_rew = ratio * adv_rew
        surr2_rew = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_rew
        loss_rew = -torch.min(surr1_rew, surr2_rew).mean()

        # Term 2: Minimize KL (The Constraint Term)
        # We use 'max' here because we are MINIMIZING this cost. 
        # If KL is high (adv_kl is positive), we want to push the ratio down.
        surr1_kl = ratio * adv_kl
        surr2_kl = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_kl
        loss_kl = torch.max(surr1_kl, surr2_kl).mean()

        return loss_rew + (beta_kl * loss_kl)
