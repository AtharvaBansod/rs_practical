import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movies dataset
data = pd.read_csv('../datasets/movie_dataset.csv')

# First, let's check the data and handle NaN values
# We'll create a content column combining relevant text data for the recommendations
data['content'] = ''

# Fill NaN values with empty strings for text columns we want to use
columns_to_use = ['overview', 'genres', 'keywords', 'tagline']
for column in columns_to_use:
    if column in data.columns:
        data[column] = data[column].fillna('')
        # For 'genres' and 'keywords' which might be in JSON format, we'll handle them
        if column in ['genres', 'keywords']:
            # Check if the column is a string (it might contain JSON data)
            if isinstance(data[column].iloc[0], str) and '{' in data[column].iloc[0]:
                # If it's JSON-like, we could extract the names, but for simplicity just use as is
                pass
        # Add the column content to our main content column
        data['content'] += ' ' + data[column]

# Clean the content column
data['content'] = data['content'].str.lower()

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Generate TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(data['content'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(movie_title, cosine_sim=cosine_sim):
    # Check if the movie title exists
    if movie_title not in data['title'].values:
        return ["Movie not found! Please try another title."]
    
    # Get the index of the movie that matches the title
    idx = data.index[data['title'] == movie_title].tolist()[0]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity score (excluding itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 10 recommendations (excluding the input movie)
    sim_scores = sim_scores[1:11]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 recommendations with their similarity scores
    recommendations = []
    for i, movie_idx in enumerate(movie_indices):
        recommendations.append(f"{i+1}. {data['title'].iloc[movie_idx]} (Score: {sim_scores[i][1]:.3f})")
    
    return recommendations

# Example usage
movie_title = "Avatar"  # Using Avatar as an example from your dataset
print(f"Recommended movies based on '{movie_title}':")
recommendations = get_recommendations(movie_title)
for rec in recommendations:
    print(rec)