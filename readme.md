# BERT4Rec Sequential Recommendation

This repository contains an end-to-end PyTorch implementation of the BERT4Rec model for sequential recommendation on the MovieLens 1M dataset.

## Repository Structure

- **`preprocess.py`**: Data preprocessing (implicit conversion, sequence generation, chronological train/val/test split).  
- **`dataloader.py`**: `TrainDataset`, `EvalDataset`, and collate functions.  
- **`model.py`**: PyTorch implementation of the BERT4Rec architecture.  
- **`train.py`**: Training loop with Adam optimizer, LR scheduling, early stopping, and seeding.  
- **`evaluate.py`**: Re-ranking evaluation with 99 negatives and multi-ground-truth metrics (Recall@10, NDCG@10).  
- **`processed_data/`**: Output directory for preprocessed `.npy` files and `metadata.json`.  
- **`model.pt`**: Saved best model checkpoint by validation NDCG@10.

## Setup

1. Clone the repository 
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```


2. install requirements.txt
   ```bash
  pip install -r requirements.txt
  ```

## Preprocessing 



## Contact

Questions? Reach out at s4628322@vuw.leidenuniv.nl