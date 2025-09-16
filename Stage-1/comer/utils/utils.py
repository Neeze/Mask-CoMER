from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from comer.datamodule.vocab import vocab
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric
import random
import math
import numpy as np

class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            pred = vocab.indices2label(pred)
            truth = vocab.indices2label(truth)

            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate
    

class MaskPredAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        pred_tokens = preds.argmax(dim=-1)
        correct = (pred_tokens == labels).sum()
        self.correct += correct
        self.total += labels.numel()

    def compute(self) -> float:
        return self.correct / self.total

class TokenMasker(nn.Module):
    def __init__(self, mask_token=-1, range_start=-1, range_end=-1):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start, range_end]
        
    def forward(self, tokens, mask_prob):
        tokens = tokens.clone()
        tokens, labels = self.perform_mask(tokens, mask_prob)
        return tokens, labels
        
    def perform_mask(self, tokens, mask_prob):
        batch_size, seq_len = tokens.size()
        device = tokens.device
        
        # Generate mask indicators
        mask_indicator = (torch.rand(tokens.size(), device=device) < mask_prob) & (tokens != 0)
        
        # Ensure at least one token is masked per sample
        no_mask = mask_indicator.sum(dim=1) == 0  # Samples with no masked tokens
        if no_mask.any():
            indices = no_mask.nonzero().squeeze(1)  # Indices of such samples
            position_idx = torch.randint(0, seq_len, (indices.size(0),), device=device)
            mask_indicator[indices, position_idx] = True
        
        labels = -torch.ones_like(tokens)
        labels[mask_indicator] = tokens[mask_indicator]
        
        # Generate probabilities for masking options
        prob = torch.rand(tokens.size(), device=device)
        rand_tokens = torch.randint(self.range[0], self.range[1], tokens.size(), device=device)
        
        # Apply masking
        mask80 = mask_indicator & (prob < 0.8)
        tokens[mask80] = self.mask_token
        
        mask10 = mask_indicator & (prob >= 0.8) & (prob < 0.9)
        tokens[mask10] = rand_tokens[mask10]
        
        # Remaining 10% keep the original token
        
        return tokens, labels

class FixedTokenMasker(TokenMasker):
    def __init__(self, mask_token=-1, range_start=-1, range_end=-1):
        super().__init__(mask_token, range_start, range_end)
        self.fixed_mask = None  
        
    def set_fixed_mask(self, tokens, mask_prob):
        batch_size, seq_len = tokens.size()
        device = tokens.device
        
        self.fixed_mask = (torch.rand(tokens.size(), device=device) < mask_prob) & (tokens != 0)
        
        no_mask = self.fixed_mask.sum(dim=1) == 0
        if no_mask.any():
            indices = no_mask.nonzero().squeeze(1)
            position_idx = torch.randint(0, seq_len, (indices.size(0),), device=device)
            self.fixed_mask[indices, position_idx] = True

    def forward(self, tokens, mask_prob=None):
        if self.fixed_mask is None:
            raise ValueError("Fixed mask has not been set. Call `set_fixed_mask` first.")
        
        tokens = tokens.clone()
        labels = -torch.ones_like(tokens)
        
        labels[self.fixed_mask] = tokens[self.fixed_mask]
        
        prob = torch.rand(tokens.size(), device=tokens.device)
        rand_tokens = torch.randint(self.range[0], self.range[1], tokens.size(), device=tokens.device)
        
        mask80 = self.fixed_mask & (prob < 0.8)
        tokens[mask80] = self.mask_token
        
        mask10 = self.fixed_mask & (prob >= 0.8) & (prob < 0.9)
        tokens[mask10] = rand_tokens[mask10]
        
        
        return tokens, labels

    
masker = TokenMasker(mask_token=vocab.MASK_IDX, range_start=3, range_end=len(vocab.word2idx.keys()))
masker_val = FixedTokenMasker(mask_token=vocab.MASK_IDX, range_start=3, range_end=len(vocab.word2idx.keys()))

def ce_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int = vocab.PAD_IDX,
    reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat[flat!=-1], flat[flat!=-1], ignore_index=ignore_idx, reduction=reduction)
    return loss


def round_down(value, decimals):
    factor = 10 ** decimals
    return math.floor(value * factor) / factor



def to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out


def mask_to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens)
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, : lens[i]] = token
        out[i, : lens[i]] = token
    return tgt, out



def mask_to_bi_tgt_out(
    tokens: List[List[int]], mask_prob: float, device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, _ = mask_to_tgt_output(tokens, "l2r", device)
    r2l_tgt, _ = mask_to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    masked_tokens, labels = masker(tgt, mask_prob)

    return masked_tokens.to(device), labels.to(device)

def mask_to_bi_tgt_out_val(
    tokens: List[List[int]], mask_prob: float, device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, _ = mask_to_tgt_output(tokens, "l2r", device)
    r2l_tgt, _ = mask_to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    masker_val.set_fixed_mask(tgt, mask_prob)
    masked_tokens, labels = masker_val(tgt)

    # print("Masked Tokens:", masked_tokens)
    # print("Labels:", labels)

    return masked_tokens.to(device), labels.to(device)