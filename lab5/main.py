# Apply support vector machines (SVMs) and neural
# networks for classification in recommender systems.



import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load dataset
# Replace this with your actual file path
data = pd.read_csv('../datasets/ratings.csv')

# Convert ratings into binary preference (liked or disliked)
data['liked'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)

# Features and labels
X = data[['userId', 'movieId']]
y = data['liked']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling (Important for SVM and Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Support Vector Machine Classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

# Neural Network Classifier
nn_model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)

# Evaluation
print("SVM Classifier Evaluation:")
print(classification_report(y_test, svm_pred))

print("Neural Network Classifier Evaluation:")
print(classification_report(y_test, nn_pred))