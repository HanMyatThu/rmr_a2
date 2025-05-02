import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from sklearn.metrics import ndcg_score

@torch.no_grad()
def evaluate(model, dataloader, device, k=10):
    model.eval()
    ndcg_scores, recall_scores = [], []

    for sequences, labels, masks in dataloader:
        sequences, labels, masks = sequences.to(device), labels.to(device), masks.to(device)
        logits = model(sequences)

        masked_positions = masks.nonzero(as_tuple=True)
        masked_logits = logits[masked_positions]
        masked_labels = labels[masked_positions]

        _, topk_indices = torch.topk(masked_logits, k=k, dim=-1)

        for preds, true_item in zip(topk_indices, masked_labels):
            relevance = np.isin(preds.cpu().numpy(), true_item.cpu().numpy()).astype(float)
            ndcg_scores.append(ndcg_score([[1] + [0]*(k-1)], [relevance], k=k))
            recall_scores.append(float(true_item in preds))

    return np.mean(ndcg_scores), np.mean(recall_scores)

if __name__ == "__main__":
    import json
    from torch.utils.data import DataLoader, TensorDataset
    from model import BERT4Rec

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = np.load('./processed_data/test.npz')
    with open('./processed_data/metadata.json', 'r') as f:
        metadata = json.load(f)

    test_dataset = TensorDataset(
        torch.LongTensor(test_data['sequences']),
        torch.LongTensor(test_data['labels']),
        torch.FloatTensor(test_data['masks'])
    )
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = BERT4Rec(
        num_items=metadata['num_items'],
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        max_seq_length=metadata['seq_length'],
        dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load('checkpoint.pt'))

    test_ndcg, test_recall = evaluate(model, test_loader, device)
    print(f"Final Test NDCG@10 = {test_ndcg:.4f}, Recall@10 = {test_recall:.4f}")