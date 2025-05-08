import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_ratings_data(n_users=1000, n_movies=500, n_ratings=100000):
    """
    Generate synthetic ratings data similar to MovieLens dataset
    
    Parameters:
    -----------
    n_users : int
        Number of unique users
    n_movies : int
        Number of unique movies
    n_ratings : int
        Total number of ratings to generate
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: userId, movieId, rating, timestamp
    """
    print(f"Generating {n_ratings} ratings for {n_users} users and {n_movies} movies...")
    
    # Generate user IDs from 1 to n_users
    user_ids = np.random.randint(1, n_users + 1, size=n_ratings)
    
    # Generate movie IDs from 1 to n_movies
    movie_ids = np.random.randint(1, n_movies + 1, size=n_ratings)
    
    # Generate ratings with a distribution similar to MovieLens
    # Based on the data, ratings are on a 0.5-5.0 scale with 0.5 increments
    rating_choices = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # Weights based on typical MovieLens rating distribution
    # Higher weights for 3-4 star ratings
    rating_weights = [0.02, 0.04, 0.02, 0.08, 0.06, 0.25, 0.15, 0.33, 0.10, 0.18]
    # Normalize weights to ensure they sum to 1
    rating_weights = [w/sum(rating_weights) for w in rating_weights]
    
    ratings = np.random.choice(rating_choices, size=n_ratings, p=rating_weights)
    
    # Generate timestamps (Unix time)
    # Using a realistic range around 2014-2015
    current_time = int(1.4e9)  # Around 2014-2015
    # Generate timestamps within a 5-year range before current_time
    timestamps = np.random.randint(current_time - 5*365*24*60*60, current_time, size=n_ratings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'userId': user_ids,
        'movieId': movie_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    return df

def main():
    # Generate the synthetic dataset
    ratings_df = generate_ratings_data(n_users=1000, n_movies=500, n_ratings=100000)
    
    # Save the synthetic data to CSV
    output_file = 'ratings.csv'
    ratings_df.to_csv(output_file, index=False)
    print(f"Saved synthetic data to '{output_file}' with {len(ratings_df)} ratings")
    
    # Basic data exploration
    print("\nData Overview:")
    print(f"Number of users: {ratings_df['userId'].nunique()}")
    print(f"Number of movies: {ratings_df['movieId'].nunique()}")
    print(f"Number of ratings: {len(ratings_df)}")
    print(f"Rating range: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")
    
    # Show first few rows
    print("\nSample of generated data:")
    print(ratings_df.head())
    
    # Rating distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings_df['rating'], bins=10, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('rating_distribution.png')
    print("Saved rating distribution visualization to 'rating_distribution.png'")
    
    # Create a binary preference column (liked or disliked)
    ratings_df['liked'] = ratings_df['rating'].apply(lambda x: 1 if x > 3 else 0)
    
    # Show class distribution
    liked_count = ratings_df['liked'].sum()
    disliked_count = len(ratings_df) - liked_count
    print(f"\nClass distribution:")
    print(f"Liked (rating > 3): {liked_count} ({liked_count/len(ratings_df)*100:.1f}%)")
    print(f"Disliked (rating ≤ 3): {disliked_count} ({disliked_count/len(ratings_df)*100:.1f}%)")

if __name__ == "__main__":
    main()