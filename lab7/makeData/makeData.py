import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker for realistic fake data
fake = Faker()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate parameters
num_users = 100
num_items = 50
num_interactions = 2000

# Generate user IDs
user_ids = [f"user_{i}" for i in range(1, num_users + 1)]

# Generate item IDs and descriptions
item_ids = [f"item_{i}" for i in range(1, num_items + 1)]
categories = ["Electronics", "Books", "Clothing", "Home", "Sports", "Beauty"]
descriptions = [fake.sentence(nb_words=10) for _ in range(num_items)]
item_features = [random.choice(categories) + " " + fake.words(nb=3)[0] for _ in range(num_items)]

# Generate interactions data
data = []
for _ in range(num_interactions):
    user_id = random.choice(user_ids)
    item_id = random.choice(item_ids)
    
    # Generate rating (1-5 stars) with some skew towards higher ratings
    rating = min(5, max(1, int(np.random.normal(4, 1))))
    
    # Generate interaction type (view, cart, purchase)
    interaction_type = random.choices(
        ["view", "cart", "purchase"],
        weights=[0.7, 0.2, 0.1],
        k=1
    )[0]
    
    # Generate timestamp (within last 6 months)
    timestamp = fake.date_time_between(start_date="-6m", end_date="now")
    
    data.append({
        "user_id": user_id,
        "item_id": item_id,
        "rating": rating,
        "interaction": interaction_type,
        "timestamp": timestamp,
        "description": descriptions[item_ids.index(item_id)],
        "item_features": item_features[item_ids.index(item_id)]
    })

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values to make it more realistic
df.loc[df.sample(frac=0.1).index, 'rating'] = np.nan
df.loc[df.sample(frac=0.05).index, 'description'] = np.nan

# Save to CSV
df.to_csv("Item_Interactions.csv", index=False)

print("Dataset created successfully with the following columns:")
print(df.columns.tolist())
print(f"\nTotal records: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Unique items: {df['item_id'].nunique()}")