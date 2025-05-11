"""
Day 8: Multicollinearity Check Using VIF
Step-by-Step:
1. Loading Dataset
2. Feature Selection:
   a. Dropped Columns that won't be used for multicollinearity check
3. Encoding Categorical Variables
4. Handling Missing Data
5. Selecting Numeric features for VIF(exclude target if needed)
6. Check for remaining NaNs or infs(infinites)
7. Standardizing features
8. Calculating VIF
9. Output Results
10. Visualization of Correlation Matrix"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Challenges/21DaysofEDA/tested.csv')
df = df.drop(['Name','Ticket','Cabin'], axis=1)

df['Sex'] = df['Sex'].map({'male':0,'female':1})
df['Embarked'] = df['Embarked'].fillna('unknown')
df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2,'unknown':3})

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

X = df.drop(['PassengerId'], axis=1)

if X.isnull().values.any() or np.isinf(X).values.any():
    print("Data contains NaNs or infs. Cleaning")
    X = X.fillna(X.meadian())
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

vif_df = pd.DataFrame()
vif_df["Feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

print("Variance Inflation factor(VIF) for Each Features: ")
print(vif_df)

plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(X_scaled, columns=X.columns).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Scaled Features")
plt.savefig("Day8.png")
plt.show()