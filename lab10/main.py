# This is main.py for lab10
# Experiment No.: 10
# Program Code (Simulated User Study Evaluation)

import random
import pandas as pd
import matplotlib.pyplot as plt

# Simulated user feedback data
user_feedback = {
    'user_id': [1, 2, 3, 4, 5],
    'recommendation_satisfaction': [random.randint(1, 5) for _ in range(5)],  # Scale 1–5
    'relevance': [random.randint(1, 5) for _ in range(5)],                   # Scale 1–5
    'usability': [random.randint(1, 5) for _ in range(5)]                    # Scale 1–5
}

# Convert to DataFrame
feedback_df = pd.DataFrame(user_feedback)

# Analyze User Feedback
def analyze_user_feedback(feedback_df):
    satisfaction_mean = feedback_df['recommendation_satisfaction'].mean()
    relevance_mean = feedback_df['relevance'].mean()
    usability_mean = feedback_df['usability'].mean()
    
    print(f"Average Recommendation Satisfaction: {satisfaction_mean:.2f}")
    print(f"Average Relevance Rating: {relevance_mean:.2f}")
    print(f"Average Usability Rating: {usability_mean:.2f}")
    
    return satisfaction_mean, relevance_mean, usability_mean

# Visualize User Feedback
def plot_feedback(feedback_df):
    plt.figure(figsize=(8, 5))
    feedback_df.drop(columns=['user_id']).mean().plot(
        kind='bar', color=['blue', 'green', 'red']
    )
    plt.title("User Study Feedback")
    plt.ylabel("Average Rating (1-5)")
    plt.xticks(rotation=45)
    plt.ylim(1, 5)  # Ratings are between 1–5
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("feedback_plot.png", bbox_inches='tight', dpi=300)
    plt.show()
    

# Run Feedback Analysis
satisfaction, relevance, usability = analyze_user_feedback(feedback_df)

# Plot Feedback Results
plot_feedback(feedback_df)