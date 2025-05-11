import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from model import BERT4Rec
from evaluate import evaluate_rerank
from dataloader import TrainDataset, EvalDataset, eval_collate_fn
from tqdm import tqdm
from logger import TrainingLogger

# set random seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.recall_value = None
        self.best_epoch = None

    def __call__(self, current_score, recall_score, epoch, model):

        # just storing for tracking (storing recall value)
        if self.recall_value is None or recall_score > self.recall_value:
            self.recall_value = recall_score

        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            self.recall_value = recall_score
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), 'model.pt')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    set_seed(42)     # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = './processed_data'
    batch_size = 32

    # Load metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)

    # Define mask token ID
    mask_token = meta['num_items'] + 1

    # Load preprocessed data
    train_inputs = np.load(os.path.join(data_dir, 'train_inputs.npy'))
    val_inputs   = np.load(os.path.join(data_dir, 'val_inputs.npy'))
    val_labels   = np.load(os.path.join(data_dir, 'val_labels.npy'), allow_pickle=True)

    # DataLoaders
    train_loader = DataLoader(
        TrainDataset(train_inputs, mask_token=mask_token, mask_prob=0.15),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        EvalDataset(val_inputs, val_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn
    )

    # Model
    model = BERT4Rec(
        num_items=meta['num_items'],
        hidden_size=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=meta['seq_length'],
        dropout=0.2
    ).to(device)

    # Optimizer & schedulers
    optimizer = AdamW(model.parameters(), lr=1e-3)
    warmup_scheduler = LambdaLR(optimizer, lambda e: min((e+1)/5, 1.0))
    main_scheduler  = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    early_stopper   = EarlyStopping(patience=5)

    # For saving training metrics
    logger = TrainingLogger()
    # Training loop
    for epoch in range(1, 101):
        model.train()
        logger.start_epoch()
        epoch_loss = 0.0
        for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        warmup_scheduler.step()
        main_scheduler.step()

        # Validation
        metrics = evaluate_rerank(
            model,
            val_loader,
            device,
            num_items=meta['num_items'],
            k=10,
            neg_samples=99
        )
        avg_loss = epoch_loss / len(train_loader)
        logger.log_epoch(epoch, avg_loss, metrics['recall'], metrics['ndcg'])
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Recall@10={metrics['recall']:.4f}, NDCG@10={metrics['ndcg']:.4f}")

        if early_stopper(metrics['ndcg'],metrics['recall'], epoch, model):
            print(f"Best Epoch: {early_stopper.best_epoch:.4f}, Best Recall@10: {early_stopper.recall_value:.4f}, Best NDCG@10: {early_stopper.best_score:.4f}")
            break

    # save log for analysis
    logger.save("training_history.csv")

if __name__ == '__main__':
    main()
