import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

df = pd.read_csv('D://Challenges/21DaysofEDA/tested.csv')

df['Age'].fillna(df['Age'].median())
df['Fare'].fillna(df['Fare'].median())
df['Embarked'].fillna(df['Embarked'].mode()[0])
# One-Hot Encoding for categorical features
df = pd.get_dummies(df, columns=['Embarked', 'Sex'], drop_first=True)

features = ['Pclass','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S']
X = df[features]
y = df['Survived']
# Check for any remaining NaNs
print("Missing values in features:\n", X.isnull().sum())

# Optionally drop rows with any NaNs (safe approach for now)
X = X.dropna()
y = y[X.index]  # match target with updated features


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

# Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Initializing the models
lr = LogisticRegression()
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
# Training models
lr.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train,y_train)


# Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Accuracy 
print(f"Logistics Regression Accuracy: {accuracy_score(y_test,y_pred_lr)}")
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Classification Report
print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_lr))
print("\nDecision Tree Report:\n", classification_report(y_test, y_pred_dt))
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))

print("Random Forest CV Accuracy:", cross_val_score(rf, X, y, cv=5).mean())
print("Train Accuracy:", rf.score(X_train, y_train))
print("Test Accuracy:", rf.score(X_test, y_test))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_rf.png")
plt.show()

# ROC Curve
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("roc_comparison.png")
plt.show()