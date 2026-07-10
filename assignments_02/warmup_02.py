import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Part 1: Warmup Exercises

# ---------------------------------------------------------------------------- #
# The scikit-learn API

# scikit-learn Question 1
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()  # create
model.fit(years, salary)  # fit
pred_4_years = model.predict(np.array([[4]]))  # predict
pred_8_years = model.predict(np.array([[8]]))  # predict

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Predicted salary for 4 years of experience: {pred_4_years[0]}")
print(f"Predicted salary for 8 years of experience: {pred_8_years[0]}\n")

# ---------------------------------------------------------------------------- #
# scikit-learn Question 2
x = np.array([10, 20, 30, 40, 50])
print(x.shape, x.ndim)
x_reshaped = x.reshape(-1, 1)
print(x_reshaped.shape, x_reshaped.ndim, "\n")

# Even when there's only one feature, Scikit-learn needs x to be 2D because it expects input data in a consistent format. Each row represents a sample and each column represents a feature.

# ---------------------------------------------------------------------------- #
# scikit-learn Question 3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)  # create
kmeans.fit(X_clusters)  # fit
labels = kmeans.predict(X_clusters)  # predict

print("Cluster Centers:", kmeans.cluster_centers_)
print("Points in each cluster:", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", marker="o")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="black",
    marker="X",
)
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("outputs/kmeans_clusters.png")

# ---------------------------------------------------------------------------- #
# Linear Regression

# data > 100 patients, each with age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker), and an annual medical cost as the target
np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Linear Regression Question 1 > create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.colorbar(label="Smoker (0=No, 1=Yes)")
plt.savefig("outputs/cost_vs_age.png")
plt.close()

# The scatter plot shows a clear separation between the two groups of patients based on smoking status. Non-smokers (blue) seem to have have lower costs across all age ranges while smokers (red points) show higher costs, suggesting that the smoker variable is effectively capturing a meaningful difference in medical costs between the groups.

# ---------------------------------------------------------------------------- #
# Linear Regression Question 2
# reshape to 2D array > 80/20 split, age (reshaped) as only feature
age_reshaped = age.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    age_reshaped, cost, test_size=0.2, random_state=42
)
print("Training features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Test target shape:", y_test.shape, "\n")

# ---------------------------------------------------------------------------- #
# Linear Regression Question 3
model = LinearRegression()
model.fit(X_train, y_train)
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_score = model.score(X_test, y_test)
print(f"RMSE: {rmse:.2f}")
print(f"R² on the test set: {r2_score:.4f}\n")

# The slope represents the average increase in medical cost for each additional year of age. In this case, for every extra year a patient ages, their medical costs are expected to increase by approximately $200 on average (assuming all other factors constant).

# ---------------------------------------------------------------------------- #
# Linear Regression Question 4 > add second feature > fit new model

X_full = np.column_stack([age, smoker])
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)
print("Age coefficient:", model_full.coef_[0])
print("Smoker coefficient:", model_full.coef_[1])

r2_score_full = model_full.score(X_full_test, y_full_test)
print(f"R² on test set with only age as feature: {r2_score:.4f}")
print(f"R² on test set with both age and smoker as features: {r2_score_full:.4f}\n")

# The smoker coefficient represents the average difference in medical cost between smokers and non-smokers, while holding age constant. For every patient who is a smoker compared to a non-smoker with the same age, their medical costs are expected to increase by approximately $15,000 on average.

# ---------------------------------------------------------------------------- #
# Linear Regression Question 5

y_full_pred = model_full.predict(X_full_test)

plt.figure(figsize=(10, 6))
plt.scatter(
    y_full_pred, y_full_test, alpha=0.7, edgecolors="k", label="Predicted vs Actual"
)

min_val = min(y_full_pred.min(), y_full_pred.min())
max_val = max(y_full_pred.max(), y_full_pred.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="sienna",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction Line",
)
plt.title("Predicted vs Actual Medical Costs")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.legend()
plt.savefig("outputs/predicted_vs_actual.png", bbox_inches="tight")
plt.close()

# Points above the diagonal represent cases where the model's prediction is higher than the actual cost (overestimations). Points below the diagonal represent cases where the model's prediction is lower than the actual cost (underestimations).
