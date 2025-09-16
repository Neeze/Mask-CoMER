import torch
import torch.nn as nn
import torch.nn.functional as F
from comer.datamodule import vocab

def focal_loss(inputs: torch.Tensor, 
               targets: torch.Tensor, 
               alpha: float = 0.25, 
               gamma: float = 2.0, 
               reduction: str = 'mean') -> torch.Tensor:
    """
    Calculate focal loss
    Args:
        inputs (torch.Tensor): [b*l, vocab_size] - Logits
        targets (torch.Tensor): [b*l] - Indexes of targets
        alpha (float): Weighting factor
        gamma (float): Focusing parameter
        reduction (str): Reduction method

    Returns:
        torch.Tensor: Loss value
    """
    probs = F.softmax(inputs, dim=1)
    targets = targets.view(-1, 1)
    probs = probs.gather(1, targets).squeeze()
    loss = -alpha * (1 - probs) ** gamma * torch.log(probs)
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()

        
def ce_loss(inputs: torch.Tensor, 
            targets: torch.Tensor,
            reduction: str = "mean") -> torch.Tensor:
    """
    Compute the cross-entropy loss
    Args:
        inputs: [b*l, vocab_size]
        targets: [b*l]
    
    Returns:
        torch.Tensor: loss value
    """
    ignore_idx = vocab.PAD_IDX
    loss = F.cross_entropy(inputs, targets, ignore_index=ignore_idx, reduction=reduction)
    return loss

