import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Titanic dataset
df = pd.read_csv('D://Challenges/21DaysofEDA/tested.csv') 

# Fill missing 'Age' with median value, 'Embarked' with the mode
df['Age'] = df['Age'].fillna(df['Age'].median())  # Avoid inplace=True warning
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Avoid inplace=True warning

# Drop 'Cabin' because it's mostly missing (or you can use 'fillna()' if you prefer)
df.drop(columns=['Cabin'], inplace=True)

# Step 4: Feature Encoding
# Convert 'Sex' and 'Embarked' to numeric values using Label Encoding or One-Hot Encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Label encoding for 'Sex'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  # One-Hot Encoding for 'Embarked'

# Step 5: Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare', 'SibSp', 'Parch']] = scaler.fit_transform(df[['Age', 'Fare', 'SibSp', 'Parch']])

# Step 6: Handle missing values in the features used for clustering
# Ensure no NaN values in the clustering columns
df[['Fare', 'SibSp', 'Parch']] = df[['Fare', 'SibSp', 'Parch']].fillna(df[['Fare', 'SibSp', 'Parch']].median())

# Step 7: Apply KMeans Clustering
# Choose the number of clusters (K) – Use the Elbow Method to determine this
# For simplicity, let's assume 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Age', 'Fare', 'SibSp', 'Parch']])

# Step 8: Visualize the clusters using PCA
# Reduce the data to 2 dimensions using PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[['Age', 'Fare', 'SibSp', 'Parch']])

# Add PCA components to the dataframe for plotting
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

# Plot the clusters in 2D using PCA components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Clusters Visualized in 2D using PCA')
plt.show()

# Step 9: Cluster Analysis & Interpretation
# Task 1: Profile the clusters (Feature distribution per cluster)
cluster_profile = df.groupby('Cluster').mean()  # or use .median() if you prefer median values
print(cluster_profile)

# Task 2: Cluster vs Survived (Examine survival rates in each cluster)
survival_by_cluster = df.groupby(['Cluster', 'Survived']).size().unstack()
print(survival_by_cluster)

# Task 3: Boxplot to see feature distribution across clusters
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Age', data=df)
plt.title('Age Distribution by Cluster')
plt.show()

# Task 4: Visualize the survival rate across clusters
plt.figure(figsize=(8, 6))
sns.barplot(x='Cluster', y='Survived', data=df, estimator=lambda x: sum(x) / len(x))
plt.title('Survival Rate by Cluster')
plt.savefig('Survival.png')
plt.show()
