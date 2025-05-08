# Analyze the functions of a recommender system using a
# real-world dataset (e.g., movie or product
# recommendations).


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
# When using the actual file, you would use:
# df = pd.read_csv('movie_dataset.csv')
# For this example, we'll create a DataFrame directly from the provided data

# Create a sample DataFrame with the data from paste.txt
# data = {
#     'title': ['Avatar'],
#     'genres': ['Action Adventure Fantasy Science Fiction']
# }
df = pd.read_csv('../datasets/movie_dataset.csv')

# Check if 'genres' column exists
if 'genres' not in df.columns:
    raise ValueError("Dataset must contain a 'genres' column for content-based filtering.")

# Fill missing values
df['genres'] = df['genres'].fillna('')

# Convert text data to numerical representation
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['genres'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Display only top 5 similar movies for each movie (shortened matrix)
top_similar_movies = pd.DataFrame(cosine_sim, index=df['title'], columns=df['title']).round(3)

# Show top 5 similar movies for each movie
print("Top 5 Similar Movies for Each Movie:")
for movie in df['title'][:5]:  # Limiting to first 5 movies to avoid long output
    top_similar = top_similar_movies[movie].nlargest(6)[1:]  # Excluding the movie itself
    print(f"\n{movie}:")
    print(top_similar)

# Movie recommender function based on most popular genre
def get_recommendations_based_on_genre(df, cosine_sim, top_n=5):
    # Count the most frequent genre(s) in the dataset
    # Handle pipe-separated genres if they exist in the dataset
    if '|' in str(df['genres'].iloc[0]):
        genre_counts = df['genres'].str.split('|').explode().value_counts()
    else:
        genre_counts = df['genres'].str.split().explode().value_counts()
    
    most_watched_genre = genre_counts.idxmax()
    
    # Filter movies with the most watched genre
    genre_filtered_df = df[df['genres'].str.contains(most_watched_genre, case=False, na=False)]
    
    # Compute recommendations based on similarity
    recommended_movies = []
    for movie in genre_filtered_df['title']:
        # Get similarity scores for the movie
        idx = df[df['title'] == movie].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies.extend(df['title'].iloc[movie_indices].tolist())
    
    # Return unique recommendations limited to top_n
    return list(dict.fromkeys(recommended_movies))[:top_n]

# Function performance evaluation based on genre
def evaluate_recommendation_based_on_genre(df, cosine_sim):
    recs = get_recommendations_based_on_genre(df, cosine_sim)
    if recs:
        print("\nTop 5 Recommended Movies from the Most Watched Genre:")
        for movie in recs:
            print(movie)
    else:
        print("No recommendations based on the most watched genre.")

# Example usage
evaluate_recommendation_based_on_genre(df, cosine_sim)