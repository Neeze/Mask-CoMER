import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
from .transforms import ScaleAugmentation, ScaleToLimitRange

from .vocab import vocab

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


class CROHMEDataset(Dataset):
    def __init__(self, 
                 ds, 
                 is_train: bool, 
                 scale_aug: bool,
                 mask_ratio: float = 0.0) -> None:
        super().__init__()
        self.ds = ds

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)

        self.mask_ratio = mask_ratio
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0.0 and 1.0")
        self.mask_generator = MaskGenerator(mask_token=vocab.MASK_IDX, 
                                            range_start=4, range_end=len(vocab.word2idx.keys()))
        
    

    def __getitem__(self, idx):
        fname, img, caption = self.ds[idx]

        img = [self.transform(im) for im in img]

        seq_ids = [vocab.words2indices(x) for x in caption]

        # add start and end tokens
        seq_ids = [[vocab.SOS_IDX] + x + [vocab.EOS_IDX] for x in seq_ids]

        # padding
        max_len = max([len(x) for x in seq_ids])
        seq_ids = [x + [0] * (max_len - len(x)) for x in seq_ids]

        indices, labels = self.mask_generator(torch.tensor(seq_ids), mask_prob=self.mask_ratio)

        return fname, img, indices, labels

    def __len__(self):
        return len(self.ds)


class MaskGenerator(nn.Module):
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
        mask_indicator = (
            (torch.rand(tokens.size(), device=device) < mask_prob)
            & (tokens != 0)
            & (tokens >= self.range[0])
            & (tokens <= self.range[1])
        )
        
        # Ensure at least one token is masked per sample
        no_mask = mask_indicator.sum(dim=1) == 0  # Samples with no masked tokens
        if no_mask.any():
            indices = no_mask.nonzero().squeeze(1)  # Indices of such samples
            position_idx = torch.randint(0, seq_len, (indices.size(0),), device=device)
            mask_indicator[indices, position_idx] = True
        
        labels = -torch.ones_like(tokens)
        labels[mask_indicator] = tokens[mask_indicator]
        
        # Apply 100% [MASK] strategy
        tokens[mask_indicator] = self.mask_token
        
        return tokens, labels