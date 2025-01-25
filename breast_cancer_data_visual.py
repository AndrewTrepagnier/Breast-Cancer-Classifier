import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the data
data = datasets.load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Create a DataFrame for easier visualization
df = pd.DataFrame(X, columns=feature_names)
df['diagnosis'] = [target_names[target] for target in y]

# 1. Basic Information
print("\n=== Dataset Overview ===")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"\nClass distribution:")
print(df['diagnosis'].value_counts())
print("\nFeature names:")
for i, name in enumerate(feature_names):
    print(f"{i+1}. {name}")

# 2. Statistical Summary
print("\n=== Statistical Summary ===")
print(df.describe())

# 3. Visualizations
plt.figure(figsize=(15, 10))

# 3.1 Class Distribution
plt.subplot(2, 2, 1)
sns.countplot(data=df, x='diagnosis')
plt.title('Distribution of Diagnoses')

# 3.2 Feature Distributions
plt.subplot(2, 2, 2)
sns.boxplot(data=df.iloc[:, :10])  # First 10 features
plt.xticks(rotation=90)
plt.title('Distribution of First 10 Features')

# 3.3 Correlation Heatmap
plt.subplot(2, 2, 3)
correlation_matrix = df.iloc[:, :-1].corr()  # Exclude diagnosis column
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlations')

# 3.4 Scatter Plot of Two Most Important Features
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='mean radius', y='mean texture', hue='diagnosis')
plt.title('Radius vs Texture by Diagnosis')

plt.tight_layout()
plt.show()

# 4. Feature Importance Analysis
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest to get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
})
importance_df = importance_df.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

# 5. Pairwise Relationships for Top Features
top_features = importance_df['feature'].head(4).tolist()
df_top = df[top_features + ['diagnosis']]

plt.figure(figsize=(12, 8))
sns.pairplot(df_top, hue='diagnosis')
plt.show()
