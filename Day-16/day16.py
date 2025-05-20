import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve

# Load Dataset
df = pd.read_csv('D://Challenges/21DaysofEDA/tested.csv')

# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Identify column types
cat_cols = ['Sex', 'Embarked']
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.difference(cat_cols).tolist()

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Pipeline
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

logreg_pipeline.fit(X_train, y_train)
y_pred_lr = logreg_pipeline.predict(X_test)

# Random Forest + GridSearchCV
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

params = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [4, 6, 8]
}

grid_search = GridSearchCV(rf_pipeline, params, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
y_pred_rf = grid_search.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_true, y_pred, model=None, proba=None):
    print(f"\n{name} Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    if proba is not None:
        roc_auc = roc_auc_score(y_true, proba)
        print(f"{name} ROC-AUC Score: {roc_auc:.4f}")

# Model Evaluations
evaluate_model("Logistic Regression", y_test, y_pred_lr, proba=logreg_pipeline.predict_proba(X_test)[:,1])
evaluate_model("Random Forest (Tuned)", y_test, y_pred_rf, proba=grid_search.predict_proba(X_test)[:,1])

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, logreg_pipeline.predict_proba(X_test)[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, grid_search.predict_proba(X_test)[:,1])

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
