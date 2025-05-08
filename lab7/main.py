# Analyze the impact of item representation methods on
# recommendation quality.




import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("../datasets/Item_Interactions.csv")

# Ensure item_id is treated as string to avoid numerical issues
data['item_id'] = data['item_id'].astype(str)
data['user_id'] = data['user_id'].astype(str)

# Create a mapping from item_id to index for consistent lookup
item_to_idx = {item_id: idx for idx, item_id in enumerate(data['item_id'].unique())}
idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}

# 1. Collaborative Filtering Representation
# Creating user-item matrix with normalized ratings/interactions
user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# Normalize per user (to account for different rating scales among users)
user_item_normalized = user_item_matrix.copy()
for user in user_item_normalized.index:
    user_ratings = user_item_normalized.loc[user]
    if user_ratings.max() > user_ratings.min():  # Avoid division by zero
        user_item_normalized.loc[user] = (user_ratings - user_ratings.min()) / (user_ratings.max() - user_ratings.min())

# Calculate item-item similarity based on user interactions
collaborative_sim = cosine_similarity(user_item_normalized.T)

# 2. Content-Based Filtering Representation
# Handle missing descriptions
if "description" in data.columns:
    # Get unique item descriptions (one per item)
    item_desc = data.drop_duplicates('item_id')[['item_id', 'description']]
    item_desc['description'] = item_desc['description'].fillna("")
    
    # Order by the same item order as collaborative filtering
    ordered_descriptions = [item_desc[item_desc['item_id'] == idx_to_item[i]]['description'].values[0] 
                           for i in range(len(idx_to_item))]
    
    # Vectorize descriptions
    content_vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    content_matrix = content_vectorizer.fit_transform(ordered_descriptions)
    content_sim = cosine_similarity(content_matrix)
else:
    # If no descriptions, use identity matrix (item only similar to itself)
    content_sim = np.eye(collaborative_sim.shape[0])

# 3. Feature-based representation (if item_features exists)
if "item_features" in data.columns:
    # Get unique item features
    item_feat = data.drop_duplicates('item_id')[['item_id', 'item_features']]
    item_feat['item_features'] = item_feat['item_features'].fillna("")
    
    # Order by the same item order as collaborative filtering
    ordered_features = [item_feat[item_feat['item_id'] == idx_to_item[i]]['item_features'].values[0] 
                       for i in range(len(idx_to_item))]
    
    # Vectorize features
    feature_vectorizer = TfidfVectorizer(token_pattern=r'[^\s]+')
    feature_matrix = feature_vectorizer.fit_transform(ordered_features)
    feature_sim = cosine_similarity(feature_matrix)
else:
    # If no features, use identity matrix
    feature_sim = np.eye(collaborative_sim.shape[0])

# 4. Hybrid approach - weighted combination of similarity matrices
# Optimized weights - can be tuned based on performance
collab_weight = 0.6
content_weight = 0.3
feature_weight = 0.1

hybrid_sim = (collab_weight * collaborative_sim + 
              content_weight * content_sim + 
              feature_weight * feature_sim)

# Function to get recommendations
def get_recommendations(item_id, similarity_matrix, top_n=5):
    if item_id not in item_to_idx:
        return []
    
    idx = item_to_idx[item_id]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first one as it's the item itself
    similar_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return [idx_to_item[idx] for idx in similar_indices]

# Evaluate recommendations
def evaluate_recommendation_quality(test_interactions, similarity_matrix, top_n=5):
    """
    Evaluates recommendation quality using precision, recall and F1 score.
    
    Args:
        test_interactions: DataFrame with user-item interactions for testing
        similarity_matrix: Item-item similarity matrix
        top_n: Number of recommendations to generate
    """
    all_recommendations = {}
    
    # Group by user to get their actual items
    user_items = test_interactions.groupby('user_id')['item_id'].apply(list).to_dict()
    
    # For each user
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for user_id, actual_items in user_items.items():
        if len(actual_items) < 2:  # Skip users with too few interactions
            continue
            
        # Use one item as "seed" and see if we can recommend the others
        seed_item = actual_items[0]
        target_items = set(actual_items[1:])
        
        if seed_item not in item_to_idx:
            continue
            
        # Get recommendations
        recommendations = set(get_recommendations(seed_item, similarity_matrix, top_n))
        
        # Calculate metrics
        true_positives = len(recommendations.intersection(target_items))
        precision = true_positives / len(recommendations) if recommendations else 0
        recall = true_positives / len(target_items) if target_items else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        
    return {
        'precision': np.mean(precision_scores) if precision_scores else 0,
        'recall': np.mean(recall_scores) if recall_scores else 0,
        'f1': np.mean(f1_scores) if f1_scores else 0
    }

# Split data into train/test (80/20)
np.random.seed(42)
msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]

# Evaluate different methods
collab_results = evaluate_recommendation_quality(test_data, collaborative_sim)
content_results = evaluate_recommendation_quality(test_data, content_sim)
hybrid_results = evaluate_recommendation_quality(test_data, hybrid_sim)

# Print results
print("\nRecommendation Quality Evaluation (Top-5):")
print(f"Collaborative Filtering: {collab_results}")
print(f"Content-Based Filtering: {content_results}")
print(f"Hybrid Approach: {hybrid_results}")

# Example recommendations
sample_item = data['item_id'].iloc[0]
print(f"\nSample Recommendations for {sample_item}:")
print(f"Collaborative: {get_recommendations(sample_item, collaborative_sim)}")
print(f"Content-Based: {get_recommendations(sample_item, content_sim)}")
print(f"Hybrid: {get_recommendations(sample_item, hybrid_sim)}")