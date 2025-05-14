import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ✅ Suppress Loky CPU count warning by manually setting physical core count
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust this to your physical core count

# Optional: Suppress all warnings (optional)
warnings.filterwarnings("ignore")

# ✅ Load dataset
df = pd.read_csv("D://Challenges/21DaysofEDA/tested.csv")

# ✅ 1. Data Preprocessing
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# ✅ Select features and scale them
features = df.select_dtypes(include=[np.number]).drop(columns=['PassengerId', 'Survived'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ✅ 2. Elbow Method to find optimal clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.show()

# ✅ 3. Apply KMeans with optimal K (e.g., K=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to DataFrame
df['Cluster'] = clusters

# ✅ 4. Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# ✅ 5. Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_components[:, 0],
    y=pca_components[:, 1],
    hue=clusters,
    palette='Set1',
    s=60
)
plt.title("Clusters Visualized using PCA (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.savefig("clusters_pca_visualization.png")
plt.show()
