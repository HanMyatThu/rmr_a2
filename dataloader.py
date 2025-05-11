import numpy as np
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    Dataset for training BERT4Rec with masked item prediction.
    Returns input sequences with random masks and corresponding labels.
    Implements BERT-style 80/10/10 masking within a 15% mask probability.
    """
    def __init__(self, sequences: np.ndarray, mask_token: int, mask_prob: float = 0.15):
        self.sequences = sequences
        self.mask_token = mask_token
        self.mask_prob = mask_prob
        self.seq_length = sequences.shape[1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Copy original sequence
        seq = self.sequences[idx].copy()
        # Initialize labels to -100 (ignore index for loss)
        labels = np.full(self.seq_length, fill_value=-100, dtype=np.int64)

        # Determine candidate positions (non-pad tokens)
        non_pad_positions = np.where(seq != 0)[0]
        # Number of positions to mask (at least one)
        num_to_mask = max(1, int(len(non_pad_positions) * self.mask_prob))
        # Randomly choose positions to mask
        mask_positions = np.random.choice(non_pad_positions, num_to_mask, replace=False)

        for pos in mask_positions:
            # Set label to original item id
            labels[pos] = seq[pos]
            rand = np.random.rand()
            if rand < 0.8:
                # 80% of the time, replace with [MASK]
                seq[pos] = self.mask_token
            elif rand < 0.9:
                # 10% of the time, replace with random item id (excluding PAD and MASK)
                # assume item ids are in [1, mask_token-1]
                seq[pos] = np.random.randint(1, self.mask_token)
            else:
                # 10% of the time, keep original item (no change)
                pass

        return torch.LongTensor(seq), torch.LongTensor(labels)


class EvalDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences
        # labels should be a numpy array of lists (dtype=object)
        self.labels = labels
        self.seq_length = sequences.shape[1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        # Each entry in labels is a list of true future item indices
        future_items = list(self.labels[idx])
        return torch.LongTensor(seq), future_items


def eval_collate_fn(batch):
    """
    Collate for EvalDataset: stacks the input tensors and gathers
    each user’s variable-length list of future_items into a Python list.
    """
    seqs = torch.stack([item[0] for item in batch], dim=0)
    future_items = [item[1] for item in batch]
    return seqs, future_items
