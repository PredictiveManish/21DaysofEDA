import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('D://Challenges/21DaysofEDA/tested.csv')

# Handle missing values
df['Cabin'] = df['Cabin'].fillna('0')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Select numeric features only (excluding ID and target)
features = df.select_dtypes(include=[np.number]).drop(['PassengerId', 'Survived'], axis=1)

# Impute missing values (if any remain) in numeric features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(features)

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X_imputed)

# Apply PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Explained variance
explained_variance = pca.explained_variance_ratio_

# Plot cumulative variance
plt.figure(figsize=(10,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.tight_layout()
plt.savefig("Cumulative_variance.png")
plt.show()


# Visualizing 2D PCA Projection
pca_2d = PCA(n_components=2)
pca_2d_features = pca_2d.fit_transform(scaled_features)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_2d_features[:, 0], y=pca_2d_features[:, 1], hue=df['Survived'])
plt.title('2D PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig('2D_projection.png')
plt.show()