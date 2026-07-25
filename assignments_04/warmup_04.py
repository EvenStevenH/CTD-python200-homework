# Part 1: Warmup Exercises
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    f1_score,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------- #
print("------ ROC Q1")

lr = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# scale the training data > KNN only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# fit to data
lr.fit(X_train, y_train)
knn.fit(X_train_scaled, y_train)

# predict probabilities on test set
y_probs_lr = lr.predict_proba(X_test)[:, 1]
y_probs_knn = knn.predict_proba(scaler.transform(X_test))[:, 1]

# AUC scores
auc_lr = roc_auc_score(y_test, y_probs_lr)
auc_knn = roc_auc_score(y_test, y_probs_knn)

print(f"Logistic Regression AUC: {auc_lr:.4f}\n" f"KNN AUC: {auc_knn:.4f}\n")

# KNN model has higher AUC (AUC=0.9394), meaning it is better at separating the positive and negative classes overall than the logistic regression model (AUC=0.7060), independent of the probability threshold chosen.

# ---------------------------------------------------------------------------- #
print("------ ROC Q2")

fpr_lr, tpr_lr, thresh_lr = roc_curve(y_test, y_probs_lr)
fpr_knn, tpr_knn, thresh_knn = roc_curve(y_test, y_probs_knn)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr, roc_auc=auc_lr).plot(
    ax=ax, name=f"Logistic Regression (AUC={auc_lr:.4f})"
)
RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn, roc_auc=auc_knn).plot(
    ax=ax, name=f"KNN (AUC={auc_knn:.4f})"
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison: Logistic Regression vs. KNN")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.close()
print("outputs/roc_comparison.png saved.\n")

# At TPR=0.8, KNN has lower FPR at 80% TPR and will have fewer false alarms. This means it's more efficient at catching 80% of positives while falsely flagging fewer negatives as positive.

# ---------------------------------------------------------------------------- #
print("------ ROC Q3")

best_f1 = -1
best_threshold = None
best_tpr = None
best_fpr = None

for threshold, tpr, fpr in zip(thresh_lr, tpr_lr, fpr_lr):
    preds = (y_probs_lr >= threshold).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold
        best_tpr = tpr
        best_fpr = fpr

print(
    f"Best Threshold: {best_threshold:.4f}\n"
    f"FPR at Optimum: {best_fpr:.4f}\n"
    f"TPR at Optimum: {best_tpr:.4f}\n"
    f"Highest F1 Score: {best_f1:.4f}\n"
)

# The optimal threshold is below 0.5. A lower threshold increases TPR at the cost of FPR, so it's useful when missing positives is more costly than false alarms. I would choose in real application, such as disease screening or fraud detection.

# ---------------------------------------------------------------------------- #
print("------ GridSearch Q1")

lr_pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
)
param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_lr = GridSearchCV(lr_pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_lr.fit(X_train, y_train)

y_probs_test = grid_lr.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs_test)

print(
    f"Best C value: {grid_lr.best_params_['clf__C']}\n"
    f"Best CV AUC: {grid_lr.best_score_:.4f}\n"
    f"Test AUC of best estimator: {test_auc:.4f}\n"
)

# Best C value from grid search was C=100.0, compared to default/guessed choice (C=1.0). This higher C value suggests that tuning improved performance, where relaxing regularization constraints was beneficial for this particular dataset and model setup.

# ---------------------------------------------------------------------------- #
print("------ GridSearch Q2")
tree_pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=42))]
)
param_grid_tree = {"clf__max_depth": [2, 3, 5, 8, None]}
gs_tree = GridSearchCV(tree_pipe, param_grid_tree, cv=5, scoring="roc_auc")
gs_tree.fit(X_train, y_train)

y_probs_test_tree = gs_tree.predict_proba(X_test)[:, 1]
test_auc_tree = roc_auc_score(y_test, y_probs_test_tree)

print(
    f"Best max_depth: {gs_tree.best_params_['clf__max_depth']}\n"
    f"Best CV AUC: {gs_tree.best_score_:.4f}\n"
    f"Test AUC (Tree): {test_auc_tree:.4f}\n"
)

# Logistic regression's AUC is lower than the decision tree's AUC, suggesting that the decision tree model performs better in distinguishing between positive and negative classes. I might bring the decision tree model into further development, but I may need to consider other factors such as interpretability, computational resources, and training time.

# ---------------------------------------------------------------------------- #
print("------ GridSearch Q3")

cv_results = grid_lr.cv_results_
mean_scores = cv_results["mean_test_score"]
std_scores = cv_results["std_test_score"]
params = grid_lr.param_grid

for idx in np.argsort(-np.array(mean_scores)):  # sort descending (best to worse)
    print(
        f"C={params['clf__C'][idx]}: Mean AUC={mean_scores[idx]:.4f} ± {std_scores[idx]:.4f}"
    )

# For cases where C value is 1.0 and 10.0, they share s similar mean score of 0.77 but slightly different standard deviations (C=10.0 has a STD 0.0057 and C=1.0 has a STD of 0.0059). Between these two I might choose C=10.0 for increased stability and consistency in performance over time or across datasets.

# ---------------------------------------------------------------------------- #
print("------ joblib Q1")

best_pipe = grid_lr.best_estimator_
best_pipe.set_params(clf__C=grid_lr.best_params_["clf__C"])
joblib.dump(best_pipe, "models/warmup_model.pkl")  # save script

loaded_clf = joblib.load("models/warmup_model.pkl")  # load script
original_preds = best_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)
assert (original_preds == loaded_preds).all(), "Predictions do not match!\n"
print("Predictions match. Model saved and loaded successfully.\n")

# If you saved only the logistic regression model (without scaler) and then called .predict(X_test) on the loaded model, where X_test is unscaled: the whole model would fail because the pipeline expects pre-scaled data, so unscaled X_test would cause errors.

# ---------------------------------------------------------------------------- #
print("------ joblib Q2")

# --- Simulated prediction script ---
model = joblib.load("models/warmup_model.pkl")
new_samples = np.array(
    [  # three hand-crafted test cases — raw, unscaled data
        [2.5, 1.2, -0.3, 0.8, 1.0, -0.5, 0.2, 0.9, -1.1, 0.4],
        [-1.0, 0.5, 0.9, -0.7, -0.2, 1.3, -0.8, 0.1, 0.5, -0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

preds = model.predict(new_samples)
probs = model.predict_proba(new_samples)
for i, (pred, prob) in enumerate(zip(preds, probs)):
    print(f"Sample {i+1}: Predicted class = {pred}, Probability = {prob[1]:.4f}")

# I expect the all-zeros row to predict based on learned weights. Without seeing data, it may be close to average prediction (near 0.5) because it contains no strong evidence for either class.
