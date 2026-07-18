# Part 1: Warmup Exercises

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# ---------------------------------------------------------------------------- #
print("------ Preprocessing Q1")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(
    f"Training set shape: {X_train.shape}\n"
    f"Target training shape: {y_train.shape}\n"
    f"Test set shape: {X_test.shape}\n"
    f"Target test shape: {y_test.shape}\n"
)

print("------ Preprocessing Q2")
# fit on training data > transform test data using same scaling parameters
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Mean of each column in X_train_scaled: {np.mean(X_train_scaled, axis=0)}\n")
# We fit the StandardScaler on X_train only to avoid data leakage and prevent the model from having access to information about the test set during training (which could lead to overly optimistic performance estimates).

# ---------------------------------------------------------------------------- #
print("------ KNN Q1")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)  # unscaled training data
y_pred_knn = knn.predict(X_test)
print(
    f"KNN Accuracy (unscaled): {accuracy_score(y_test, y_pred_knn)}\n"
    f"Classification Report: {classification_report(y_test, y_pred_knn)}\n"
)

print("------ KNN Q2")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
print(f"KNN Accuracy (scaled): {accuracy_score(y_test, y_pred_knn_scaled)}\n")
# Scaling does not significantly improve performance for this dataset because KNN relies on distance calculations between points. Scaling can sometimes distort the relative distances in feature space. As all features are already on a similar scale (0-4), the impact of scaling is minimal.

print("------ KNN Q3")
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print(
    f"CV Scores: {cv_scores}\n"
    f"Mean CV Score: {np.mean(cv_scores):.4f}\n"
    f"Standard Deviation: {np.std(cv_scores):.4f}\n"
)
# Cross-validation results are more trustworthy than a single train/test split because they provide a more robust estimate of model performance by averaging over multiple folds. This reduces the impact of random chance in the data split.

print("------ KNN Q4")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = -np.inf

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    cv_scores_k = cross_val_score(knn_k, X_train, y_train, cv=5)  # 5 fold
    mean_score = np.mean(cv_scores_k)
    print(f"k={k}, Mean CV Score: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"Best k: {best_k} with a score of {best_score:.4f}\n")
# The optimal k value is likely 5, as it provides a good balance between bias and variance. Higher k values can lead to under-fitting (smoother decision boundaries), while lower k values can lead to overfitting (noisy decision boundaries).

# ---------------------------------------------------------------------------- #
print("------ Classifier Evaluation Q1")
cm = confusion_matrix(y_test, y_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for KNN on Iris Dataset")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()
print("knn_confusion_matrix.png saved.\n")
# The model most often confuses setosa with versicolor and virginica. This is expected because these species are more similar to each other in the feature space than they are to setosa.

# ---------------------------------------------------------------------------- #
# The sklearn API: Decision Trees
print("------ Decision Trees Q1")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(
    f"Decision Tree Accuracy (unscaled): {accuracy_score(y_test, y_pred_dt)}\n"
    f"Classification Report: {classification_report(y_test, y_pred_dt)}\n"
)
# Decision Trees generally perform well on this dataset with an accuracy around 95-97% because they can capture non-linear relationships and interactions between features without requiring distance calculations. Scaling does not affect their performance because they do not rely on distance metrics like KNN. The tree structure is built based on splits in feature values, not their relative scales.

# ---------------------------------------------------------------------------- #
# Logistic Regression and Regularization
# use "saga", as "liblinear" only supports binary classification
print("------ Logistic Regression Q1")
for C in [0.01, 1.0, 100]:
    model = LogisticRegression(C=C, max_iter=1000, solver="liblinear")
    model.fit(X_train_scaled, y_train)
    total_coef_magnitude = np.abs(model.coef_).sum()
    print(f"C={C}, Total Coefficient Magnitude: {total_coef_magnitude:.4f}\n")
# As C increases, the total coefficient magnitude also increases. A larger C value corresponds to weaker regularization (higher penalty for large coefficients), allowing the model to fit more complex decision boundaries. Regularization helps prevent overfitting by constraining the model's complexity.

# ---------------------------------------------------------------------------- #
# PCA
digits = load_digits()
X_digits = digits.data  # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images = digits.images  # same data shaped as 8x8 images for plotting

print("------ PCA Q1")
print(f"Shape of X_digits: {X_digits.shape}\n" f"Shape of images: {images.shape}\n")
selected_indices = []
for label in range(10):  # select ine image per digit before plotting
    idx = np.where(y_digits == label)[0][0]
    selected_indices.append(idx)
plt.figure(figsize=(10, 2))
for i, idx in enumerate(selected_indices):
    plt.subplot(1, 10, i + 1)
    plt.imshow(images[idx], cmap="gray_r")
    plt.title(f"Digit {y_digits[idx]}")
    plt.axis("off")
plt.savefig("outputs/sample_digits.png")
plt.close()
print("sample_digits.png saved.\n")

print("------ PCA Q2")
pca = PCA()
scores = pca.fit_transform(X_digits)
plt.figure(figsize=(8, 6))
plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(label="Digit")  # c = color array
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection of Digits")
plt.savefig("outputs/pca_2d_projection.png")
plt.close()
print("pca_2d_projection.png saved.\n")
# Same-digit images will tend to cluster together in this 2D space because PCA captures the directions of maximum variance in the data. In other words, similar digits will have similar patterns and thus be close in this reduced-dimensional space.

print("------ PCA Q3")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs. Number of Components")
plt.grid()
plt.savefig("outputs/pca_variance_explained.png")
plt.close()
print("pca_variance_explained.png saved.\n")
# Approximately 40-60 components are needed to explain 80% of the variance, depending on the exact cumulative sum at that point.


print("------ PCA Q4")
n_values = [2, 5, 15, 40]
sample_indices = range(5)


def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction += scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


fig, axes = plt.subplots(len(n_values) + 1, len(sample_indices), figsize=(8, 6))
for i, idx in enumerate(sample_indices):  # original row
    axes[0, i].imshow(images[idx], cmap="gray_r")
    axes[0, i].set_title(f"Original ({y_digits[idx]})")
    axes[0, i].axis("off")
for j, n in enumerate(n_values):  # reconstructed rows
    row_idx = i + 1
    for k, idx in enumerate(sample_indices):
        reconstructed = reconstruct_digit(idx, scores, pca, n)
        axes[row_idx, j].imshow(reconstructed, cmap="gray")
        axes[row_idx, j].set_title(f"n={n}")
        axes[row_idx, j].axis("off")
plt.suptitle("PCA Reconstructions of Digits")
plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()
print("pca_reconstructions.png saved.\n")
# Digits become clearly recognizable at around n=15. This matches where the variance curve levels off, indicating that 15 components capture most of the structure in the data.
