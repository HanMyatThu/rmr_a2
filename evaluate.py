import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import BERT4Rec
from dataloader import EvalDataset, eval_collate_fn

# set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def evaluate_rerank(
    model, dataloader, device, num_items, k=10, neg_samples=99
):
    model.eval()
    items = np.arange(1, num_items + 1)
    recalls, ndcgs = [], []
    pad_token = model.pad_token
    mask_token = model.mask_token

    with torch.no_grad():
        for seqs, future_items_list in dataloader:
            seqs = seqs.to(device)
            batch_size, seq_len = seqs.size()

            # Mask the last non-pad position for each sequence
            masked_seqs = seqs.clone()
            mask_positions = []
            for i in range(batch_size):
                non_pad = torch.nonzero(seqs[i] != pad_token, as_tuple=True)[0]
                pos = non_pad[-1].item() if len(non_pad) > 0 else 0
                masked_seqs[i, pos] = mask_token
                mask_positions.append(pos)

            # Forward pass
            logits = model(masked_seqs)
            # Extract logits at masked positions
            scores = logits[torch.arange(batch_size), mask_positions, :].cpu().numpy()

            for i, future_items in enumerate(future_items_list):
                if len(future_items) == 0:
                    continue

                # Evaluate only the immediate next item
                true_item = future_items[0]
                # Negative sampling excluding user history
                user_history = set(seqs[i].cpu().numpy().tolist())
                neg_pool = [x for x in items if x not in user_history]
                negs = np.random.choice(neg_pool, neg_samples, replace=False)

                # Candidates: one positive + negatives
                candidates = np.concatenate([[true_item], negs])
                candidate_scores = scores[i, candidates]
                rank_indices = np.argsort(-candidate_scores)
                top_k = candidates[rank_indices][:k]

                # Recall@k
                recall = 1.0 if true_item in top_k else 0.0
                recalls.append(recall)

                # NDCG@k
                if true_item in top_k:
                    rank = int(np.where(top_k == true_item)[0][0])
                    ndcg = 1.0 / math.log2(rank + 2)
                else:
                    ndcg = 0.0
                ndcgs.append(ndcg)

    return {'recall': np.mean(recalls), 'ndcg': np.mean(ndcgs)}

if __name__ == '__main__':
    data_dir = './processed_data'
    checkpoint = 'model.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)

    # Load test data
    test_inputs = np.load(os.path.join(data_dir, 'test_inputs.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'), allow_pickle=True)

    test_loader = DataLoader(
        EvalDataset(test_inputs, test_labels),
        batch_size=64,
        shuffle=False,
        collate_fn=eval_collate_fn
    )

    # Reload model with same hyperparameters used during training
    model = BERT4Rec(
        num_items=meta['num_items'],
        hidden_size=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=meta['seq_length'],
        dropout=0.2
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    # Run evaluation
    metrics = evaluate_rerank(
        model,
        test_loader,
        device,
        num_items=meta['num_items'],
        k=10,
        neg_samples=99
    )
    print(f"Test Recall@10: {metrics['recall']:.4f}, Test NDCG@10: {metrics['ndcg']:.4f}")