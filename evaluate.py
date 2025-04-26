import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split

# Load embeddings and labels
with open("Data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
with open("Data/names.pkl", "rb") as f:
    names = pickle.load(f)

embeddings = np.array(embeddings)
names = np.array(names)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, names, test_size=0.3, random_state=42)

# Settings
THRESHOLD = 0.6  # Same threshold you use in recognize_faces
y_true = []
y_pred = []

# Evaluate
for i in range(len(X_test)):
    true_label = y_test[i]
    test_emb = X_test[i]

    min_dist = float('inf')
    predicted_label = "Unknown"

    for j in range(len(X_train)):
        dist = cosine(test_emb, X_train[j])
        if dist < THRESHOLD and dist < min_dist:
            min_dist = dist
            predicted_label = y_train[j]

    y_true.append(true_label)
    y_pred.append(predicted_label)

# Convert Unknowns to special label
y_true_binary = [1 if label != "Unknown" else 0 for label in y_true]
y_pred_binary = [1 if label != "Unknown" else 0 for label in y_pred]

# Calculate metrics
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)

# Calculate FAR and FRR
false_accepts = sum((np.array(y_true) != np.array(y_pred)) & (np.array(y_pred) != "Unknown"))
false_rejects = sum((np.array(y_true) != np.array(y_pred)) & (np.array(y_pred) == "Unknown"))

FAR = false_accepts / len(y_true) * 100
FRR = false_rejects / len(y_true) * 100

# Print results
print("===== Evaluation Metrics =====")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"False Acceptance Rate (FAR): {FAR:.2f}%")
print(f"False Rejection Rate (FRR): {FRR:.2f}%")
print("================================")
