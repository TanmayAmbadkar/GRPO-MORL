import torch
import torch.nn.functional as F

def get_batch_logps(logits, labels, attention_mask):
    """
    Safe log-prob calculation with padding masking.
    Clamps labels to valid vocabulary range to prevent CUDA index errors.
    """
    vocab_size = logits.size(-1)
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Clamp labels to valid range [0, vocab_size - 1]
    shift_labels = shift_labels.clamp(0, vocab_size - 1)
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    shift_mask = attention_mask[..., 1:].contiguous()
    return (per_token_logps * shift_mask).sum(dim=1)

def compute_grpo_advantages(tensor):
    """Calculates Group-Relative Advantage: (x - mean) / std."""
    # tensor shape: [Batch, Group_Size] or [Group_Size]
    # If [Group_Size], we treat it as Batch=1
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    return (tensor - mean) / (std + 1e-8)
