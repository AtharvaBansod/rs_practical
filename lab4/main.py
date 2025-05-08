import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("../datasets/movie_dataset.csv")

# Define classification target (Hit = 1, Flop = 0)
median_revenue = df['revenue'].median()
df['success'] = np.where((df['revenue'] > median_revenue) & (df['vote_average'] > 6.5), 1, 0)

# Select features
features = ['budget', 'popularity', 'vote_average', 'vote_count', 'runtime']
df_selected = df[features + ['success']].dropna()

# Print dataset stats
print(f"Dataset shape: {df_selected.shape}")
print(f"Success distribution: \n{df_selected['success'].value_counts()}")
print(f"Success rate: {df_selected['success'].mean():.2f}")

# Split dataset
X = df_selected[features]
y = df_selected['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifiers
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=42)
rule_based = DecisionTreeClassifier(random_state=42, max_depth=3)  # Rule-based using decision tree with limited depth
rf = RandomForestClassifier(random_state=42, n_estimators=100)  # Adding Random Forest classifier

# Fit models
knn.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)
rule_based.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred_knn = knn.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test_scaled)
y_pred_rule = rule_based.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)

# Evaluation
print("\nKNN Classifier:")
print(classification_report(y_test, y_pred_knn))

print("\nDecision Tree Classifier:")
print(classification_report(y_test, y_pred_dt))

print("\nRule-Based Classifier:")
print(classification_report(y_test, y_pred_rule))

print("\nRandom Forest Classifier:")
print(classification_report(y_test, y_pred_rf))

# Cross-validation scores
cv_scores_knn = cross_val_score(knn, X_train_scaled, y_train, cv=5)
cv_scores_dt = cross_val_score(dt, X_train_scaled, y_train, cv=5)
cv_scores_rule = cross_val_score(rule_based, X_train_scaled, y_train, cv=5)
cv_scores_rf = cross_val_score(rf, X_train_scaled, y_train, cv=5)

print("\nCross-validation mean accuracy:")
print(f"KNN: {cv_scores_knn.mean():.4f}")
print(f"Decision Tree: {cv_scores_dt.mean():.4f}")
print(f"Rule-based: {cv_scores_rule.mean():.4f}")
print(f"Random Forest: {cv_scores_rf.mean():.4f}")

# Print extracted rules
rules = export_text(rule_based, feature_names=features)
print("\nRule-Based Classifier Decision Rules:")
print(rules)

# Feature importance for the random forest model
feature_importance = pd.DataFrame(
    {'Feature': features, 'Importance': rf.feature_importances_}
).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance)

# Example movie recommendation
def predict_success(movie_features, classifier=rf, scaler=scaler):
    """
    Predict if a movie will be a success based on its features
    
    Parameters:
    movie_features (dict): Dictionary with features: budget, popularity, vote_average, 
                           vote_count, runtime
    classifier: The trained classifier to use for prediction
    scaler: The fitted scaler to normalize input features
    
    Returns:
    prediction (int): 1 for Hit, 0 for Flop
    probability (float): Probability of success
    """
    # Create a dataframe from the input features
    features_df = pd.DataFrame([movie_features])
    
    # Scale the features
    scaled_features = scaler.transform(features_df)
    
    # Make prediction
    prediction = classifier.predict(scaled_features)[0]
    
    # Get probability if classifier supports it
    try:
        probability = classifier.predict_proba(scaled_features)[0][1]
    except:
        probability = None
        
    return prediction, probability

# Example usage
example_movie = {
    'budget': 150000000,  # $150M budget
    'popularity': 100,    # Popularity score
    'vote_average': 7.5,  # Average rating
    'vote_count': 1000,   # Number of votes
    'runtime': 120        # 2 hour runtime
}

prediction, probability = predict_success(example_movie)
print("\nExample Movie Prediction:")
print(f"Features: {example_movie}")
print(f"Prediction: {'Hit' if prediction == 1 else 'Flop'}")
if probability is not None:
    print(f"Success Probability: {probability:.2f}")