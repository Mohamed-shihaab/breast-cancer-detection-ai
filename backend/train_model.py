import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

print("Loading extracted features...")

X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=8, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("PCA reduced shape:", X_train.shape)

# 🔥 GRID SEARCH SVM
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf"]
}

svm = SVC(class_weight="balanced", probability=True)

grid = GridSearchCV(
    svm,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=2
)

print("Training SVM with GridSearch...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# Evaluation
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
))

# Save everything
joblib.dump(best_model, "svm_model.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Optimized SVM model saved")
