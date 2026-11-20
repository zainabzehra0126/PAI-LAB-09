import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Class 0 (malignant):", np.sum(y == 0))
print("Class 1 (benign):", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Test accuracy:", round(model.score(X_test, y_test), 4))

probs = model.predict_proba(X_test)[:, 1]
for t in [0.3, 0.5, 0.7]:
    pred = (probs >= t).astype(int)
    print("Threshold", t)
    print("Predicted class 1:", pred.sum())
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

scores = model.decision_function(X_test)
manual = sigmoid(scores)
print("Sigmoid matches predict_proba:", np.allclose(manual, probs))
print("Maximum difference:", np.max(np.abs(manual - probs)))