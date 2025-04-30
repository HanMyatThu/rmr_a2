import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
'''
# convert all dat to csv files
# ratings
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'])

ratings.to_csv('ml-1m/ratings.csv', index=False)

# users
users = pd.read_csv('ml-1m/users.dat', sep='::', engine='python',
                    names=['userId', 'gender', 'age', 'occupation', 'zipCode'])

users.to_csv('ml-1m/users.csv', index=False)

# movies
movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                     names=['movieId', 'title', 'genres'], encoding='ISO-8859-1')

movies.to_csv('ml-1m/movies.csv', index=False)

print("All Dat files are converted to csv")
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_movielens(data_path, output_path='edited_ratings.csv', seq_length=20):
    # 1. Load the MovieLens dataset
    df = pd.read_csv(data_path)  # Assumes rating.csv with columns: userId, movieId, rating, timestamp
    
    # 2. Treat df ≥ 4 as positive interactions, ignore df < 4
    df = df[df['rating'] >= 4].copy()  # Filter out df < 4
    df['implicit'] = 1  # All remaining interactions are positive (implicit feedback = 1)
    
    # 3. Convert each user’s history into a chronologically ordered sequence
    df = df.sort_values(by=['userId', 'timestamp'])
    
    # 4. Filter out users with fewer than 5 interactions
    user_counts = df.groupby('userId').size()
    valid_users = user_counts[user_counts >= 5].index
    df = df[df['userId'].isin(valid_users)]
    
    # 5. Save the processed df to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Processed df saved to {output_path}")
    
    # 6. Split dataset into training, validation, and testing by user
    users = df['userId'].unique()
    train_users, temp_users = train_test_split(users, train_size=0.7, random_state=42)
    val_users, test_users = train_test_split(temp_users, train_size=0.5, random_state=42)
    
    train_data = df[df['userId'].isin(train_users)]
    val_data = df[df['userId'].isin(val_users)]
    test_data = df[df['userId'].isin(test_users)]
    
    # 7. Perform sequence truncation or padding to a fixed length
    def create_sequences(data, seq_length):
        sequences = []
        labels = []
        for user in data['userId'].unique():
            user_data = data[data['userId'] == user][['movieId', 'implicit']].values
            if len(user_data) > seq_length:
                # Truncate to the most recent seq_length interactions
                user_data = user_data[-seq_length:]
            elif len(user_data) < seq_length:
                # Pad with zeros (assuming 0 is not a valid movieId)
                padding = np.zeros((seq_length - len(user_data), 2), dtype=int)
                user_data = np.vstack([padding, user_data])
            
            # Input sequence is all but the last interaction, label is the last interaction
            sequences.append(user_data[:-1, 0])  # Movie IDs for sequence
            labels.append(user_data[-1, 1])      # Implicit feedback for last interaction
        return np.array(sequences), np.array(labels)
    
    # Generate sequences for train, val, and test sets
    train_sequences, train_labels = create_sequences(train_data, seq_length)
    val_sequences, val_labels = create_sequences(val_data, seq_length)
    test_sequences, test_labels = create_sequences(test_data, seq_length)
    
    return {
        'train': {'sequences': train_sequences, 'labels': train_labels},
        'val': {'sequences': val_sequences, 'labels': val_labels},
        'test': {'sequences': test_sequences, 'labels': test_labels}
    }

# Example usage
data_path = 'ml-1m/ratings.csv'  # Update with actual path
preprocessed_data = preprocess_movielens(data_path)

# Output shapes for verification
print("Train sequences shape:", preprocessed_data['train']['sequences'].shape)
print("Validation sequences shape:", preprocessed_data['val']['sequences'].shape)
print("Test sequences shape:", preprocessed_data['test']['sequences'].shape)