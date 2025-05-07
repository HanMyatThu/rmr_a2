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

# use re-ranking evaluation strategy: with negative sampling 99
def evaluate_rerank(
    model, dataloader, device, num_items, k=10, neg_samples=99
):
    model.eval()
    items = np.arange(1, num_items + 1)
    recalls, ndcgs = [], []

    with torch.no_grad():
        for seqs, future_items_list in dataloader:
            seqs = seqs.to(device)
            logits = model(seqs)
            scores = logits[:, -1, :].cpu().numpy()  # use last position predictions

            for i, future_items in enumerate(future_items_list):
                if len(future_items) == 0:
                    continue

                # Build negative pool excluding user history
                user_history = set(seqs[i].cpu().numpy().tolist())
                neg_pool = [x for x in items if x not in user_history]
                negs  = np.random.choice(neg_pool, neg_samples, replace=False)

                # candidatges: true items + negatives
                users = np.concatenate([np.array(future_items), negs])
                users_scores = scores[i, users]
                rank_indices = np.argsort(-users_scores)
                top_k_users = users[rank_indices][:k]

                # recall: percentage of true items in top k !!
                hits    = set(future_items) & set(top_k_users.tolist())
                recall  = len(hits) / len(future_items)
                recalls.append(recall)

                # NDCG
                dcg = 0.0
                for rank, item in enumerate(top_k_users):
                    if item in hits:
                        dcg += 1.0 / math.log2(rank + 2)
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(future_items), k)))
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

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
    
    # debug issue on test
    train_inputs = np.load(os.path.join(data_dir, 'train_inputs.npy'))
    test_items = set(np.concatenate(test_labels))
    train_items = set(np.unique(train_inputs))
    print(f"Test items not in training: {len(test_items - train_items)}")

    test_loader = DataLoader(
        EvalDataset(test_inputs, test_labels),
        batch_size=64,
        shuffle=False,
        collate_fn=eval_collate_fn
    )

    # reload model with same hyperparameters used during training
    model = BERT4Rec(
        num_items=meta['num_items'],
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        max_seq_len=meta['seq_length'],
        dropout=0.2
    )

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
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

