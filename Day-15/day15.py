import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv('D://Challenges/21DaysofEDA/tested.csv')

# ----------------------------
# Data Cleaning
# ----------------------------
# Fix warning-prone assignments
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop columns not useful for ML model input
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Target and features
y = df['Survived']
X_base = df.drop('Survived', axis=1)

# ----------------------------
# 1. Target Encoding
# ----------------------------
X_target = X_base.copy()
target_enc = ce.TargetEncoder()
X_target[['Sex', 'Embarked']] = target_enc.fit_transform(X_target[['Sex', 'Embarked']], y)

# ----------------------------
# 2. Frequency Encoding
# ----------------------------
X_freq = X_base.copy()
for col in ['Sex', 'Embarked']:
    freq = X_freq[col].value_counts()
    X_freq[col] = X_freq[col].map(freq)

# ----------------------------
# 3. Ordinal Encoding
# ----------------------------
X_ord = X_base.copy()
ord_enc = OrdinalEncoder()
X_ord[['Sex', 'Embarked']] = ord_enc.fit_transform(X_ord[['Sex', 'Embarked']])

# ----------------------------
# 4. One-Hot Encoding
# ----------------------------
X_ohe = pd.get_dummies(X_base, columns=['Sex', 'Embarked'], drop_first=True)

# ----------------------------
# Train & Evaluate Function
# ----------------------------
def train_and_evaluate(X, y, method_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{method_name} Encoding Accuracy: {acc:.4f}")
    return acc

# ----------------------------
# Run All Encodings
# ----------------------------
accuracies = {
    'Target': train_and_evaluate(X_target, y, 'Target'),
    'Frequency': train_and_evaluate(X_freq, y, 'Frequency'),
    'Ordinal': train_and_evaluate(X_ord, y, 'Ordinal'),
    'One-Hot': train_and_evaluate(X_ohe, y, 'One-Hot')
}

# ----------------------------
# Accuracy Comparison Graph
# ----------------------------
plt.figure(figsize=(9,6))
plt.bar(accuracies.keys(), accuracies.values(), color='lightseagreen')
plt.title('Accuracy Comparison of Encoding Techniques')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("accuracy.png")
plt.show()
