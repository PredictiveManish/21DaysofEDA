import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset
df = pd.read_csv('D://Challenges/21DaysofEDA/tested.csv')

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# One-Hot Encoding for categorical features
df = pd.get_dummies(df, columns=['Embarked', 'Sex'], drop_first=True)

# Select relevant columns for clustering
df = df[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked_Q', 'Embarked_S', 'Sex_male']]

# Scaling the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=df)
plt.title('KMeans Clustering - Titanic Dataset')
plt.savefig('Kmeans.jpg')
plt.show()

# Profiling clusters - Cluster vs Age
sns.boxplot(x='Cluster', y='Age', data=df)
plt.title('Cluster vs Age')
plt.savefig('Clusterage.png')
plt.show()

# Profiling clusters - Cluster vs Fare
sns.boxplot(x='Cluster', y='Fare', data=df)
plt.title('Cluster vs Fare')
plt.savefig('clusterfare.png')
plt.show()

# Visualizing survival in each cluster
sns.countplot(x='Cluster', hue='Survived', data=df)
plt.title('Cluster vs Survival')
plt.savefig('clustersurvived.png')
plt.show()
