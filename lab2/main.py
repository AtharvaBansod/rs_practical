# Compare different recommendation techniques
# (content-based, collaborative filtering, hybrid).



# This is main.py for lab2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('../datasets/movie_dataset.csv')

# Check required columns
required_columns = {'title', 'genres', 'director', 'popularity', 'vote_average', 'vote_count'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain {required_columns} columns.")

# Extract director from crew column if necessary
if 'director' not in df.columns and 'crew' in df.columns:
    # Extract director names from the crew column
    def get_director(crew_data):
        try:
            if crew_data:
                return crew_data.split('James Cameron')[0].strip()
            return ''
        except:
            return ''
    
    df['director'] = df['crew'].apply(get_director)

# Fill missing values
df['genres'] = df['genres'].fillna('')
df['director'] = df['director'].fillna('')
df['features'] = df['genres'] + ' ' + df['director']

# ---------------------- Content-Based Filtering ----------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_based_recommendations(title, df, cosine_sim, top_n=5):
    # Create a mapping of movie titles to their indices
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    # Get the index of the movie that matches the title
    idx = indices.get(title)
    
    if idx is None:
        return []
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar movies
    return df['title'].iloc[movie_indices].tolist()

# ---------------------- Collaborative Filtering ----------------------
# Use popularity, vote_average, and vote_count as collaborative features
collab_df = df[['popularity', 'vote_average', 'vote_count']].fillna(0)

# Scale features for better SVD performance
scaler = StandardScaler()
collab_df_scaled = scaler.fit_transform(collab_df)

# Apply SVD - matrix factorization technique
svd = TruncatedSVD(n_components=3, random_state=42)
collab_matrix = svd.fit_transform(collab_df_scaled)

# Compute similarity between movies
collab_cosine_sim = cosine_similarity(collab_matrix)

def get_collaborative_recommendations(title, df, collab_cosine_sim, top_n=5):
    # Create a mapping of movie titles to their indices
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    # Get the index of the movie that matches the title
    idx = indices.get(title)
    
    if idx is None:
        return []
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(collab_cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top N most similar movies
    return df['title'].iloc[movie_indices].tolist()

# ---------------------- Hybrid Recommendation System ----------------------
def get_hybrid_recommendations(title, df, cosine_sim, collab_cosine_sim, top_n=5):
    # Get recommendations from both systems
    content_recs = get_content_based_recommendations(title, df, cosine_sim, top_n)
    collab_recs = get_collaborative_recommendations(title, df, collab_cosine_sim, top_n)
    
    # Combine recommendations and remove duplicates
    hybrid_recs = list(set(content_recs + collab_recs))
    
    # Return top N recommendations
    return hybrid_recs[:top_n]

# ---------------------- Performance Comparison ----------------------
def evaluate_recommendation_system(title, df, cosine_sim, collab_cosine_sim):
    print(f"\nRecommendations for '{title}':")
    print(f"Content-Based: {get_content_based_recommendations(title, df, cosine_sim)}")
    print(f"Collaborative Filtering: {get_collaborative_recommendations(title, df, collab_cosine_sim)}")
    print(f"Hybrid: {get_hybrid_recommendations(title, df, cosine_sim, collab_cosine_sim)}")

# ---------------------- Example Usage ----------------------
# Find a movie title that actually exists in the dataset
# If 'Cars 2' is not in the dataset, use the first movie title as an example
if 'Cars 2' in df['title'].values:
    movie_title = 'Cars 2'
else:
    movie_title = df['title'].iloc[0]  # Use first movie in dataset as example

print(f"Using movie: {movie_title}")
evaluate_recommendation_system(movie_title, df, cosine_sim, collab_cosine_sim)