import numpy as np
import json
from collections import defaultdict

with open('./processed_data/metadata.json', 'r') as f:
    meta = json.load(f)

item2idx = meta['item2idx']
pad_token = meta['pad_token']
seq_length = meta['seq_length']
num_items = meta['num_items']

# Load all data files
train_inputs = np.load('./processed_data/train_inputs.npy', allow_pickle=True)
val_inputs = np.load('./processed_data/val_inputs.npy', allow_pickle=True)
test_inputs = np.load('./processed_data/test_inputs.npy', allow_pickle=True)
val_labels = np.load('./processed_data/val_labels.npy', allow_pickle=True)
test_labels = np.load('./processed_data/test_labels.npy', allow_pickle=True)
train_users = np.load('./processed_data/train_users.npy', allow_pickle=True)
val_users = np.load('./processed_data/val_users.npy', allow_pickle=True)
test_users = np.load('./processed_data/test_users.npy', allow_pickle=True)


def validate_splits():
    # Ensure all users are unique and matched across splits
    assert len(set(train_users) & set(val_users) & set(test_users)) == len(train_users), "User mismatch in splits!"
    
    # Check sequence lengths and labels length
    print(f"\nSequence lengths (train/val/test): {len(train_inputs)}, {len(val_inputs)}, {len(test_inputs)}")
    print(f"Labels lengths (val/test): {len(val_labels)}, {len(test_labels)}")

def inspect_sequences(user_id=None):
    # check input/labels for a specific user or random user 
    idx = np.random.choice(len(train_users)) if user_id is None else np.where(train_users == user_id)[0][0]
    user = train_users[idx]
    
    print(f"\nInspecting user {user}:")
    print(f"Train input (padded): {train_inputs[idx]}")
    print(f"Val input (padded):   {val_inputs[idx]}")
    print(f"Test input (padded):  {test_inputs[idx]}")
    print(f"Val labels (raw):     {val_labels[idx]}")
    print(f"Test labels (raw):    {test_labels[idx]}")

    # Check if val/test labels are in the item vocabulary
    for label in val_labels[idx] + test_labels[idx]:
        assert label in item2idx.values(), f"Label {label} not in item2idx!"

# check all sequence has value of 20
def check_padding():
    for seq in train_inputs:
        assert len(seq) == seq_length, f"Sequence length mismatch: {len(seq)} != {seq_length}"
    
    print("\nPadding/truncation validated: All sequences are length", seq_length)

def check_negative_samples():
    #  Ensure ground-truth items don't appear in negative samples 
    user_items = defaultdict(set)
    for user, seq in zip(train_users, train_inputs):
        user_items[user].update(seq)
    
    # Check val/test labels against user history
    for user, labels in zip(val_users, val_labels):
        for label in labels:
            assert label not in user_items[user], f"Val label {label} leaked into user {user}'s history!"
    
    for user, labels in zip(test_users, test_labels):
        for label in labels:
            assert label not in user_items[user], f"Test label {label} leaked into user {user}'s history!"
    
    print("\nNegative sampling check passed: No ground-truth items in user history.")

if __name__ == '__main__':
    print("DEBUGGING PROCESSED DATA")
    print(f"Metadata: {meta}")
    
    validate_splits()
    inspect_sequences()
    check_padding()
    check_negative_samples()
    
    print("\nAll checks passed!")