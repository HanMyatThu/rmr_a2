import os
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from train import set_seed
from model import BERT4Rec
from evaluate import evaluate_rerank
from dataloader import TrainDataset, EvalDataset, eval_collate_fn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# Directory for ablation results
os.makedirs('ablation', exist_ok=True)

# Default hyperparameters for training and architecture
DEFAULTS = {
    'batch_size': 64,
    'dropout': 0.2,
    'hidden_size': 128,
    'learning_rate': 5e-4,
    'num_layers': 2,
    'neg_samples': 99, # For evaluation part within run_train_eval
    'epochs': 5,
    'num_heads': 4, # Default num_heads, can be made ablative if needed
    'mask_prob': 0.15 # Default mask_prob for TrainDataset
}

# Values to try for each hyperparameter
ABLATION_VALUES = {
    'batch_size': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3],
    'hidden_size': [64, 128, 256],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'num_layers': [1, 2, 3],
    'neg_samples': [50, 99, 150],
    'epochs': [3, 5, 8]
}

# Load data and metadata
with open('./processed_data/metadata.json', 'r') as f:
    meta = json.load(f)
train_inputs = np.load('./processed_data/train_inputs.npy')
val_inputs = np.load('./processed_data/val_inputs.npy')
val_labels = np.load('./processed_data/val_labels.npy', allow_pickle=True)

mask_token = meta['num_items'] + 1
seq_length = meta['seq_length']
num_items = meta['num_items']

def run_train_eval(hparams):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run_start_time = time.time()

    # DataLoaders
    train_loader = DataLoader(
        TrainDataset(train_inputs, mask_token=mask_token, mask_prob=hparams.get('mask_prob', DEFAULTS['mask_prob'])),
        batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(
        EvalDataset(val_inputs, val_labels),
        batch_size=hparams['batch_size'], shuffle=False, collate_fn=eval_collate_fn)
    
    # Model
    model = BERT4Rec(
        num_items=num_items,
        hidden_size=hparams['hidden_size'],
        num_heads=hparams.get('num_heads', DEFAULTS['num_heads']),
        num_layers=hparams['num_layers'],
        max_seq_len=seq_length,
        dropout=hparams['dropout']
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=hparams['learning_rate'])
    warmup_epochs = max(1, hparams['epochs'] // 10)
    total_steps = len(train_loader) * hparams['epochs']
    
    def lr_lambda(current_step):
        if current_step < len(train_loader) * warmup_epochs:
            return float(current_step) / float(max(1, len(train_loader) * warmup_epochs))
        return 1.0
        
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch in range(hparams['epochs']):
        model.train()
        epoch_loss = 0
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            warmup_scheduler.step()
            epoch_loss += loss.item()

    model.eval() 
    metrics = evaluate_rerank(
        model, 
        val_loader,
        device, 
        num_items=num_items, 
        k=10, 
        neg_samples=hparams['neg_samples']
    )
    
    total_run_time = time.time() - run_start_time
    return metrics['recall'], metrics['ndcg'], total_run_time

def ablation_study():
    results = []
    
    for hp_name, hp_values in ABLATION_VALUES.items():
        print(f"\nAblating {hp_name}...")
        for val in hp_values:
            hparams = DEFAULTS.copy()
            hparams[hp_name] = val
            
            for key in DEFAULTS:
                if key not in hparams:
                    hparams[key] = DEFAULTS[key]

            recall, ndcg, run_time = run_train_eval(hparams)
            print(f"{hp_name}={val}: Recall@10={recall:.4f}, NDCG@10={ndcg:.4f}, Time={run_time:.1f}s")
            results.append({
                'hyperparameter': hp_name,
                'value': val,
                'recall': recall,
                'ndcg': ndcg,
                'run_time': run_time
            })
            
    df = pd.DataFrame(results)
    df.to_csv('ablation/ablation_results_full_training.csv', index=False)
    plot_ablation(df)

def plot_ablation(df):
    import pandas as pd
    for hp_name in ABLATION_VALUES.keys():
        sub_df = df[df['hyperparameter'] == hp_name]
        if sub_df.empty:
            print(f"No data found for hyperparameter: {hp_name}. Skipping plot.")
            continue
        
        best_row = sub_df.loc[sub_df['ndcg'].idxmax()]
        
        default_val_for_hp = DEFAULTS[hp_name]
        default_perf_rows = sub_df[sub_df['value'] == default_val_for_hp]
        
        if default_perf_rows.empty:
            print(f"Default value {default_val_for_hp} for hyperparameter {hp_name} not found in results for plotting. Using overall default.")
            default_ndcg = DEFAULTS.get(f'{hp_name}_default_ndcg', 0)
            default_value_to_display = default_val_for_hp
            if not sub_df[sub_df['value'] == default_val_for_hp].empty:
                 default_ndcg = sub_df[sub_df['value'] == default_val_for_hp].iloc[0]['ndcg']
        else:
            default_ndcg = default_perf_rows.iloc[0]['ndcg']
            default_value_to_display = default_perf_rows.iloc[0]['value']

        plt.figure(figsize=(6,3)) 
        plt.bar(['Default', 'Best'], [default_ndcg, best_row['ndcg']], color=['blue', 'red'])
        plt.title(f'{hp_name}: Default ({default_value_to_display}) vs Best ({best_row["value"]})')
        plt.ylabel('NDCG@10')
        plt.savefig(f'ablation/{hp_name}_default_vs_best_full_training.png')
        plt.close()
        
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    plotted_something = False
    for hp_name in ABLATION_VALUES.keys():
        sub_df = df[df['hyperparameter'] == hp_name]
        if sub_df.empty:
            continue
        axs[0].scatter(sub_df['run_time'], sub_df['recall'], label=hp_name)
        axs[1].scatter(sub_df['run_time'], sub_df['ndcg'], label=hp_name)
        plotted_something = True
        
    if plotted_something:
        axs[0].set_xlabel('Run Time (s)') 
        axs[0].set_ylabel('Recall@10')
        axs[0].set_title('Run Time vs Recall@10') 
        axs[1].set_xlabel('Run Time (s)') 
        axs[1].set_ylabel('NDCG@10')
        axs[1].set_title('Run Time vs NDCG@10') 
        axs[0].legend()
        axs[1].legend()
    else:
        print("No data to plot for time vs metric scatter plot.")

    plt.tight_layout()
    plt.savefig('ablation/time_vs_metric_full_training.png')
    plt.close()

if __name__ == '__main__':
    ablation_study() 