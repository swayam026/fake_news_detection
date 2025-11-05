# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load datasets ---
true = pd.read_csv("data/True.csv")
fake = pd.read_csv("data/Fake.csv")

# --- Add labels ---
true["label"] = "REAL"
fake["label"] = "FAKE"

# --- Combine and shuffle ---
data = pd.concat([true, fake]).sample(frac=1, random_state=42).reset_index(drop=True)
data["text"] = data["title"].astype(str) + " " + data["text"].astype(str)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, stratify=data["label"], random_state=42
)

# --- Pipeline ---
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5)),
    ("clf", LogisticRegression(max_iter=1000))
])

# --- Train ---
print("\nTraining model...")
model.fit(X_train, y_train)

# --- Predict ---
y_pred = model.predict(X_test)

# --- Metrics ---
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# --- Save model ---
joblib.dump(model, "model.pkl")
print("\n✅ Model saved as model.pkl")

# --- Plot Confusion Matrix ---
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Confusion matrix saved as confusion_matrix.png")
