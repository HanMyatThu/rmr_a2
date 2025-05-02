import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

# Constants
MASK_TOKEN = 0
SEQ_LENGTH = 20
DATA_PATH = './dataset/ratings.dat'
OUTPUT_DIR = './processed_data'

def convert_to_csv(input_path, output_path):
    """Convert MovieLens .dat file to CSV with proper headers."""
    df = pd.read_csv(input_path, sep='::', engine='python', 
                    names=['userId', 'movieId', 'rating', 'timestamp'])
    df.to_csv(output_path, index=False)
    print(f"Converted {input_path} to {output_path}")

def create_sequences(data, seq_length, mask_prob=0.2):
    """
    Creates sequences with random masking for BERT4Rec.
    Args:
        data: DataFrame containing user-item interactions
        seq_length: Fixed sequence length
        mask_prob: Probability of masking an item
    Returns:
        sequences: Padded/truncated sequences with masks
        labels: Ground truth for masked items
        masks: Positions of masked items (1=masked, 0=not masked)
    """
    sequences, labels, masks = [], [], []
    
    for user in data['userId'].unique():
        items = data[data['userId'] == user]['movieIdx'].values
        
        # Pad/truncate
        if len(items) > seq_length:
            items = items[-seq_length:]
        else:
            items = np.pad(items, (seq_length - len(items), 0), 
                          'constant', constant_values=MASK_TOKEN)
        
        # Create masked sequence
        masked_seq = items.copy()
        mask_positions = np.random.choice(seq_length, size=int(seq_length * mask_prob), 
                         replace=False)
        
        # Store labels only for masked positions
        label_seq = np.full(seq_length, MASK_TOKEN)  # Default ignore index
        label_seq[mask_positions] = items[mask_positions]
        
        # Apply masking
        masked_seq[mask_positions] = MASK_TOKEN
        
        sequences.append(masked_seq)
        labels.append(label_seq)
        masks.append((masked_seq == MASK_TOKEN).astype(int))  # Binary mask
    
    return np.array(sequences), np.array(labels), np.array(masks)

# def create_eval_sequences(data, seq_length):
#     sequences, labels, masks = [], [], []
    
#     for user in data['userId'].unique():
#         items = data[data['userId'] == user]['movieIdx'].values
#         if len(items) > seq_length:
#             items = items[-seq_length:]
#         else:
#             items = np.pad(items, (seq_length - len(items), 0), 'constant', constant_values=MASK_TOKEN)
        
#         # Always mask the last token
#         label_seq = np.full(seq_length, MASK_TOKEN)
#         label_seq[-1] = items[-1]
        
#         masked_seq = items.copy()
#         masked_seq[-1] = MASK_TOKEN  # Replace last item with [MASK]
        
#         mask = np.zeros(seq_length)
#         mask[-1] = 1
        
#         sequences.append(masked_seq)
#         labels.append(label_seq)
#         masks.append(mask)
    
#     return np.array(sequences), np.array(labels), np.array(masks)


def preprocess(data_path, output_dir=OUTPUT_DIR):
    """Full preprocessing pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and filter data
    df = pd.read_csv(data_path)
    df = df[df['rating'] >= 4].sort_values(['userId', 'timestamp'])
    
    # 2. Filter inactive users
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df = df[df['userId'].isin(valid_users)]
    
    # 3. Create movieID mapping
    movie_to_idx = {movie: idx+1 for idx, movie in enumerate(df['movieId'].unique())}
    df['movieIdx'] = df['movieId'].map(movie_to_idx)
    
    # 4. Train/val/test split
    users = df['userId'].unique()
    train_users, temp_users = train_test_split(users, test_size=0.3, random_state=42)
    val_users, test_users = train_test_split(temp_users, test_size=0.5, random_state=42)
    
    # 5. Create sequences
    train_data = df[df['userId'].isin(train_users)]
    val_data = df[df['userId'].isin(val_users)]
    test_data = df[df['userId'].isin(test_users)]
    
    train_sequences, train_labels, train_masks = create_sequences(train_data, SEQ_LENGTH)
    val_sequences, val_labels, val_masks = create_sequences(val_data, SEQ_LENGTH)
    test_sequences, test_labels, test_masks = create_sequences(test_data, SEQ_LENGTH)
    
    # 6. Save everything
    np.savez_compressed(
        os.path.join(output_dir, 'train.npz'),
        sequences=train_sequences,
        labels=train_labels,
        masks=train_masks
    )
    np.savez_compressed(
        os.path.join(output_dir, 'val.npz'),
        sequences=val_sequences,
        labels=val_labels,
        masks=val_masks
    )
    np.savez_compressed(
        os.path.join(output_dir, 'test.npz'),
        sequences=test_sequences,
        labels=test_labels,
        masks=test_masks
    )
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'movie_to_idx': {int(k): int(v) for k, v in movie_to_idx.items()},  # Convert keys/values to int
            'num_items': int(len(movie_to_idx)),  # Ensure this is native int
            'mask_token': int(MASK_TOKEN),
            'seq_length': int(SEQ_LENGTH)
        }, f)
    print(f"Preprocessing complete. Files saved to {output_dir}")

def main():
    # Convert if needed
    if not os.path.exists('./dataset/ratings.csv'):
        convert_to_csv(DATA_PATH, './dataset/ratings.csv')
    
    # Run preprocessing
    preprocess('./dataset/ratings.csv')

if __name__ == "__main__":
    main()