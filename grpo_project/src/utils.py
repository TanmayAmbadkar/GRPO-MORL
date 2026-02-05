import torch
import torch.nn.functional as F
import numpy as np

def get_batch_logps(logits, labels, attention_mask, label_pad_token_id=-100):
    """
    Computes log probabilities for labels, strictly masking padding.
    Input:
        logits: [Batch, Seq_Len, Vocab]
        labels: [Batch, Seq_Len]
        attention_mask: [Batch, Seq_Len]
    """
    # Shift so that tokens < n predict n
    # logits[..., :-1, :] predicts labels[..., 1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate log_softmax
    # [Batch, Seq_Len-1, Vocab]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs of the actual labels
    # [Batch, Seq_Len-1, 1]
    per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Create mask (ignore padding in labels)
    # We must shift the mask to align with the labels
    shift_mask = attention_mask[..., 1:].contiguous()
    
    # Zero out logps for padding tokens
    per_token_logps = per_token_logps * shift_mask
    
    # Sum over the sequence length
    return per_token_logps.sum(dim=1)

def sample_weights(batch_size, num_objectives=4, alpha=1.0, device="cpu"):
    """
    Samples dynamic preference weights.
    Includes 20% 'Corner Case' logic (focus on single objective).
    """
    if np.random.rand() < 0.2:
        # Corner case: One-hot vector
        indices = torch.randint(0, num_objectives, (batch_size,))
        weights = torch.zeros(batch_size, num_objectives)
        weights[torch.arange(batch_size), indices] = 1.0
    else:
        # Dirichlet distribution for mixed preferences
        dist = torch.distributions.Dirichlet(torch.ones(num_objectives) * alpha)
        weights = dist.sample((batch_size,))
    
    return weights.to(device)

def compute_grouped_advantages(rewards, eps=1e-8):
    """
    GRPO Normalization: (r - mean(r)) / std(r) within the group.
    Input: [Batch, Group_Size, Num_Objectives]
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean) / (std + eps)
    return advantages
