# day6_outlier_removal.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load Titanic dataset (replace path if needed)
df = pd.read_csv('D:/Challenges/21DaysofEDA/tested.csv')

# Show original shape
print(f"Original DataFrame shape: {df.shape}")

# ========== Z-SCORE METHOD ==========
# Calculate Z-scores for numerical columns
z_scores = np.abs(zscore(df[['Age', 'Fare']].dropna()))
df_z = df[['Age', 'Fare']].dropna()
df_z_clean = df_z[(z_scores < 3).all(axis=1)]

# Merge back with original dataframe
df_no_outliers = df.loc[df_z_clean.index]

print(f"Shape after Z-score outlier removal: {df_no_outliers.shape}")

# ========== IQR METHOD (Alternative) ==========
# Calculate IQR for Age and Fare
Q1 = df[['Age', 'Fare']].quantile(0.25)
Q3 = df[['Age', 'Fare']].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df_iqr = df[~((df[['Age', 'Fare']] < (Q1 - 1.5 * IQR)) | (df[['Age', 'Fare']] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"Shape after IQR outlier removal: {df_iqr.shape}")

# ========== VISUALIZATION ==========
# Boxplots before and after outlier removal
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df[['Age', 'Fare']], palette='Set3')
plt.title('Before Outlier Removal')

plt.subplot(1, 2, 2)
sns.boxplot(data=df_no_outliers[['Age', 'Fare']], palette='Set2')
plt.title('After Z-score Outlier Removal')
plt.tight_layout()
plt.savefig('outlier_removal_comparison.png')
plt.show()

# ========== SAVE CLEANED DATA ==========
df_no_outliers.to_csv('titanic_cleaned_zscore.csv', index=False)
df_iqr.to_csv('titanic_cleaned_iqr.csv', index=False)

print("Cleaned datasets saved as 'titanic_cleaned_zscore.csv' and 'titanic_cleaned_iqr.csv'")
