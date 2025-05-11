import pandas as pd
import numpy as np
import json
import os

PAD_TOKEN = 0
SEQ_LENGTH = 20
DATA_PATH = './dataset/ratings.dat'
OUTPUT_DIR = './processed_data'

# Pad or truncate a sequence to fixed length (20)
def padOrTruncate(seq, max_len=SEQ_LENGTH):
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [PAD_TOKEN] * (max_len - len(seq)) + seq


def preprocess(data_path: str = DATA_PATH, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # load and filter positive interactions (rating >= 4)
    df = pd.read_csv(
        data_path,
        sep=r'::',
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp']
    )
    df = df[df.rating >= 4].copy()

    # generate chronological interaction sequences per user.
    df.sort_values(['userId', 'timestamp'], inplace=True)
    userSequence = df.groupby('userId')['movieId'].apply(list).to_dict()

    # filter out users with fewer than 5 interactions
    userSequence = {u: seq for u, seq in userSequence.items() if len(seq) >= 5}

    # map movie IDs (0 reserved for PAD)
    all_items = sorted({item for seq in userSequence.values() for item in seq})
    item2idx = {item: idx + 1 for idx, item in enumerate(all_items)}
    num_items = len(item2idx)

    train_inputs, val_inputs, test_inputs = [], [], []
    val_labels, test_labels = [], []
    train_users, val_users, test_users = [], [], []

    # split each user's sequence into train/val/test and prepare contexts & labels
    for user, seq in userSequence.items():
        # remap items
        seq = [item2idx[i] for i in seq]
        n = len(seq)
        i1 = int(n * 0.7)
        i2 = int(n * 0.85)

        # Training: use first i1 items as input
        train_seq = seq[:i1]
        train_input = padOrTruncate(train_seq)

        # Validation: input is prefix up to i1, labels are seq[i1:i2]
        val_input = padOrTruncate(seq[:i1])
        val_label = seq[i1:i2]

        # Test: input is prefix up to i2, labels are seq[i2:]
        test_input = padOrTruncate(seq[:i2])
        test_label = seq[i2:]

        train_inputs.append(train_input)
        val_inputs.append(val_input)
        test_inputs.append(test_input)
        val_labels.append(val_label)
        test_labels.append(test_label)
        train_users.append(user)
        val_users.append(user)
        test_users.append(user)

    # save inputs and labels
    np.save(os.path.join(output_dir, 'train_inputs.npy'), np.array(train_inputs, dtype=np.int32))
    np.save(os.path.join(output_dir, 'val_inputs.npy'),   np.array(val_inputs,   dtype=np.int32))
    np.save(os.path.join(output_dir, 'test_inputs.npy'),  np.array(test_inputs,  dtype=np.int32))
    np.save(os.path.join(output_dir, 'val_labels.npy'),   np.array(val_labels,   dtype=object))
    np.save(os.path.join(output_dir, 'test_labels.npy'),  np.array(test_labels,  dtype=object))
    np.save(os.path.join(output_dir, 'train_users.npy'),  np.array(train_users))
    np.save(os.path.join(output_dir, 'val_users.npy'),    np.array(val_users))
    np.save(os.path.join(output_dir, 'test_users.npy'),   np.array(test_users))

    # save metadata
    meta = {
        'item2idx':   item2idx,
        'num_items':  num_items,
        'pad_token':  PAD_TOKEN,
        'seq_length': SEQ_LENGTH
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)

    print(f"Preprocessing complete. Processed data saved to {output_dir}")

if __name__ == '__main__':
    preprocess()
