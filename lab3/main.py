import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv('../datasets/movie_dataset.csv')

# ------------------- 1. Handling Missing Values -------------------
# Fill missing numerical values with median
numerical_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Filling missing categorical values with 'Unknown'
categorical_cols = ['genres', 'original_language', 'director']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# ------------------- 2. Encoding Categorical Variables -------------------
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# ------------------- 3. Normalizing Ratings -------------------
scaler = MinMaxScaler()
df[['vote_average', 'vote_count', 'popularity']] = scaler.fit_transform(df[['vote_average', 'vote_count', 'popularity']])

# ------------------- 4. Feature Selection -------------------
selected_features = ['title', 'genres', 'original_language', 'popularity', 'vote_average', 'vote_count', 'director']
df_selected = df[selected_features]

# ------------------- 5. Display Processed Dataset (First 15 Rows) -------------------
print("Processed Dataset (First 15 Rows):")
print(df_selected.head(15))