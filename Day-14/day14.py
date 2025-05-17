import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np

df = pd.read_csv('D://Challenges/21DaysofEDA/tested.csv')

df = df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Separate target
y = df['Survived']
X = df.drop('Survived', axis=1)

# Identify column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Define preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Get feature names after OneHotEncoding
cat_encoded_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
feature_names = list(num_cols) + list(cat_encoded_names)

# Convert to DataFrame for analysis
X_df = pd.DataFrame(X_processed, columns=feature_names)
X_df['Survived'] = y.values  # Add target back temporarily for visualizations

# -----------------------------
# 1. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(14, 10))
sns.heatmap(X_df.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("Correlation.png")
plt.show()

# -----------------------------
# 2. Pairplot on selected features
# -----------------------------
pairplot_cols = ['Age', 'Fare', 'Pclass', 'Survived']
if all(col in X_df.columns for col in pairplot_cols):
    sns.pairplot(X_df[pairplot_cols], hue='Survived', palette='Set2')
    plt.suptitle('Pairwise Feature Relationships', y=1.02)
    plt.savefig("pairplot.png")
    plt.show()
else:
    print("Pairplot columns not all found after encoding.")

# -----------------------------
# 3. t-SNE Visualization
# -----------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(X_processed)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=y, palette='cool')
plt.title('t-SNE Visualization of Clusters')
plt.tight_layout()
plt.savefig("tsne.png")
plt.show()

# -----------------------------
# 4. UMAP Visualization
# -----------------------------
reducer = umap.UMAP(random_state=42)
umap_result = reducer.fit_transform(X_processed)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=y, palette='Set1')
plt.title('UMAP Visualization of Clusters')
plt.tight_layout()
plt.savefig("umap.png")
plt.show()