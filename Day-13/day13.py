import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D://Challenges/21DaysofEDA/tested.csv')

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# Drop columns that won't be used
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

# Encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Logistic Regression Tuning
# ------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_param = {'C': [0.01, 0.1, 1, 10, 100]}
log_grid = GridSearchCV(log_reg, log_param, cv=5)
log_grid.fit(X_train, y_train)

print("Best Logistic Regression Params:", log_grid.best_params_)
log_best = log_grid.best_estimator_
log_pred = log_best.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

# ------------------------------
# Decision Tree Tuning
# ------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt_param = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
dt_grid = GridSearchCV(dt, dt_param, cv=5)
dt_grid.fit(X_train, y_train)

print("Best Decision Tree Params:", dt_grid.best_params_)
dt_best = dt_grid.best_estimator_
dt_pred = dt_best.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# ------------------------------
# Random Forest Tuning
# ------------------------------
rf = RandomForestClassifier(random_state=42)
rf_param = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf, rf_param, cv=5)
rf_grid.fit(X_train, y_train)

print("Best Random Forest Params:", rf_grid.best_params_)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ------------------------------
# Feature Importance (Random Forest)
# ------------------------------
importances = rf_best.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.tight_layout()
plt.savefig("feature.png")
plt.show()
