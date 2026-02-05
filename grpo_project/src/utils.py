# src/utils.py
import torch
import torch.nn.functional as F
import numpy as np

def get_batch_logps(logits, labels, attention_mask, label_pad_token_id=-100):
    """
    Calculates the log probabilities of the given labels under the logits.
    CRITICAL: Properly masks padding so the model doesn't learn to predict 'pad'.
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate log_softmax
    # [Batch, Seq_Len-1, Vocab]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs of the actual labels
    # shift_labels: [Batch, Seq_Len-1] -> [Batch, Seq_Len-1, 1]
    per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding
    # attention_mask is usually 1 for data, 0 for pad.
    # We shift it to match labels.
    shift_mask = attention_mask[..., 1:].contiguous()
    
    # Zero out logps for padding tokens
    per_token_logps = per_token_logps * shift_mask
    
    # Sum over the sequence length to get sequence log prob
    return per_token_logps.sum(dim=1)

def sample_weights(batch_size, num_objectives=4, alpha=1.0, device="cpu"):
    """
    Samples preference weights from a Dirichlet distribution.
    Includes 'Corner Cases' (1.0 on one objective) to force distinct behaviors.
    """
    # 20% chance of hard corner (focus purely on 1 objective)
    if np.random.rand() < 0.2:
        indices = torch.randint(0, num_objectives, (batch_size,))
        weights = torch.zeros(batch_size, num_objectives)
        weights[torch.arange(batch_size), indices] = 1.0
    else:
        # Dirichlet sampling for mixed preferences
        dist = torch.distributions.Dirichlet(torch.ones(num_objectives) * alpha)
        weights = dist.sample((batch_size,))
    
    return weights.to(device)

def compute_grouped_advantages(rewards, eps=1e-8):
    """
    GRPO Advantage Calculation: (r - mean(r)) / std(r)
    Input: [Batch, Group_Size, Num_Objectives]
    Output: [Batch, Group_Size, Num_Objectives]
    """
    # Mean and Std calculated over the Group dimension (dim=1)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    
    # Normalize
    advantages = (rewards - mean) / (std + eps)
    return advantages
