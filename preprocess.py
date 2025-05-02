import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

# Constants
PAD_TOKEN = 0
SEQ_LENGTH = 20
DATA_PATH = './dataset/ratings.dat'
OUTPUT_DIR = './processed_data'

# Helper to build fixed-length sequences
def build_sequences(df):
    seqs = []
    for user in sorted(df['userId'].unique()):
        items = df[df['userId'] == user].sort_values('timestamp')['movieIdx'].values
        if len(items) > SEQ_LENGTH:
            seq = items[-SEQ_LENGTH:]
        else:
            seq = np.pad(
                items,
                (SEQ_LENGTH - len(items), 0),
                'constant',
                constant_values=PAD_TOKEN
            )
        seqs.append(seq)
    return np.array(seqs)


def preprocess(data_path: str = DATA_PATH, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and filter explicit ratings >= 4
    df = pd.read_csv(
        data_path,
        sep='::', engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp']
    )
    df = df[df['rating'] >= 4].copy()

    # 2. Filter out users with fewer than 5 interactions
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df = df[df['userId'].isin(valid_users)]

    # 3. Create dense movie ID mapping
    movie_to_idx = {movie: idx + 1 for idx, movie in enumerate(df['movieId'].unique())}
    df['movieIdx'] = df['movieId'].map(movie_to_idx)
    num_items = len(movie_to_idx)

    # 4. Split users into train/val/test
    users = df['userId'].unique()
    train_users, temp_users = train_test_split(users, test_size=0.3, random_state=42)
    val_users, test_users = train_test_split(temp_users, test_size=0.5, random_state=42)

    # Save user splits
    # np.save(os.path.join(output_dir, 'train_users.npy'), train_users)
    # np.save(os.path.join(output_dir, 'val_users.npy'), val_users)
    # np.save(os.path.join(output_dir, 'test_users.npy'), test_users)

    # Build and save sequences
    train_seqs = build_sequences(df[df['userId'].isin(train_users)])
    val_seqs   = build_sequences(df[df['userId'].isin(val_users)])
    test_seqs  = build_sequences(df[df['userId'].isin(test_users)])

    np.save(os.path.join(output_dir, 'train_sequences.npy'), train_seqs)
    np.save(os.path.join(output_dir, 'val_sequences.npy'),   val_seqs)
    np.save(os.path.join(output_dir, 'test_sequences.npy'),  test_seqs)

    meta = {
        'movie_to_idx': {int(k): int(v) for k, v in movie_to_idx.items()},
        'num_items': num_items,
        'pad_token': PAD_TOKEN,
        'seq_length': SEQ_LENGTH
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)

    print(f"Preprocessing complete. Data saved to {output_dir}")


if __name__ == '__main__':
    preprocess()
