# Conduct offline experiments to test recommendation
# algorithms using historical data.



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("../datasets/Item_Interactions.csv", usecols=['user_id', 'item_id', 'rating'])

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create user-item matrix
train_matrix = train_data.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# Fit Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
model.fit(train_matrix.values)

# Prediction function
def make_predictions(user_id, item_id):
    if user_id not in train_matrix.index or item_id not in train_matrix.columns:
        return np.nan
    
    user_idx = train_matrix.index.get_loc(user_id)
    distances, indices = model.kneighbors(train_matrix.iloc[user_idx].values.reshape(1, -1), n_neighbors=5)
    
    similar_users = train_matrix.iloc[indices.flatten()]
    return similar_users[item_id].mean()

# Evaluation function
def evaluate_model():
    actual = []
    predicted = []
    
    for _, row in test_data.iterrows():
        pred = make_predictions(row['user_id'], row['item_id'])
        if not np.isnan(pred):
            actual.append(row['rating'])
            predicted.append(pred)
    
    if not actual:  # If no valid predictions
        return np.nan, np.nan, np.nan
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))
    
    actual_bin = [1 if r >= 4 else 0 for r in actual]
    pred_bin = [1 if r >= 4 else 0 for r in np.round(predicted)]
    
    precision = precision_score(actual_bin, pred_bin, zero_division=0)
    recall = recall_score(actual_bin, pred_bin, zero_division=0)
    
    return rmse, precision, recall

# Run evaluation
rmse, precision, recall = evaluate_model()

# Print results
print("Evaluation Results:")
print(f"RMSE: {rmse:.4f}" if not np.isnan(rmse) else "RMSE: No valid predictions")
print(f"Precision: {precision:.4f}" if not np.isnan(precision) else "Precision: No valid predictions")
print(f"Recall: {recall:.4f}" if not np.isnan(recall) else "Recall: No valid predictions")