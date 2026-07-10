import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ---------------------------------------------------------------------------- #
# Task 1: Load and Explore > pre-preprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "student_performance_math.csv")
df = pd.read_csv(path, sep=";")  # use semicolon separator

print("Shape of dataset:", df.shape)
print("First 5 rows:\n", df.head())
print("Data types:\n", df.dtypes, "\n")

# Plot histogram of G3 with 21 bins (0-20 grades)
plt.figure(figsize=(10, 6))
plt.hist(df["G3"], bins=21, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Grade (0-20)")
plt.ylabel("Frequency")
plt.savefig("outputs/g3_distribution.png")

# ---------------------------------------------------------------------------- #
# Task 2: Preprocess the Data

df2 = df[df["G3"] != 0].copy()
print("Shape (before):", df.shape)
print("Shape (after):", df2.shape)
# filter out G3=0 rows > prevent biasing target variable (G3), because they aren't representative of actual student performance

yes_no_columns = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_columns:
    df2[col] = df2[col].map({"yes": 1, "no": 0})
df2["sex"] = df2["sex"].map({"F": 0, "M": 1})

corr_before = df["absences"].corr(df["G3"], method="pearson")
corr_after = df2["absences"].corr(df2["G3"], method="pearson")  # removed G3=0 rows
print(f"Pearson correlation between absences and G3 (before): {corr_before:.2f}")
print(f"Pearson correlation between absences and G3 (after): {corr_after:.2f}\n")
# filtering changes result > absences are not correlated with a real score, and can weaken relationship between absences and G3

# ---------------------------------------------------------------------------- #
# Task 3: Exploratory Data Analysis

numeric_features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
    "schoolsup",
    "internet",
    "higher",
    "activities",
    "sex",
]
correlations = df2[numeric_features + ["G3"]].corr()["G3"].sort_values()
print("Correlation of numeric features with G3 (sorted):")
print(correlations, "\n")
# "failures" > strongest negative relationship > as it increases, predicted grade decreases
# "higher" (wanting to pursue higher education) > strongest positive relationship

plt.figure(figsize=(10, 6))
plt.scatter(df2["absences"], df2["G3"])
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/g3_vs_absences.png")
plt.close()
# students with lower absence counts tend to have high grades

fig, ax = plt.subplots(figsize=(10, 6))
failure_groups = [
    df2[df2["failures"] == f]["G3"].values for f in sorted(df2["failures"].unique())
]
failure_labels = [str(f) for f in sorted(df2["failures"].unique())]
ax.boxplot(failure_groups, tick_labels=failure_labels)
ax.set_title("G3 Distribution by Number of Past Failures")
ax.set_xlabel("Past Failures")
ax.set_ylabel("G3 (Final Grade)")
fig.savefig("outputs/g3_by_failures.png", bbox_inches="tight")
plt.close(fig)
# students with 0 past failures have a higher grade distribution
# additional failures shift the median down > strong predictor

fig, ax = plt.subplots(figsize=(10, 6))
groups = [df2[df2["higher"] == v]["G3"].values for v in [0, 1]]
ax.boxplot(groups, tick_labels=["No (0)", "Yes (1)"])
ax.set_title("G3 Distribution by Higher-Education Aspiration")
ax.set_xlabel("Wants Higher Education")
ax.set_ylabel("G3 (Final Grade)")
fig.savefig("outputs/g3_by_higher.png", bbox_inches="tight")
plt.close(fig)
# students wanting to pursue higher education score higher on average

# ---------------------------------------------------------------------------- #
# Task 4: Baseline Model > use failures alone to predict G3

# split data into training and test sets > fit baseline model using only "failures"
X_baseline = df2[["failures"]].values
y_baseline = df2["G3"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_baseline, y_baseline, test_size=0.2, random_state=42
)

model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)

y_pred_baseline = model_baseline.predict(X_test)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_baseline = r2_score(y_test, y_pred_baseline)
print(f"Baseline Model Slope: {model_baseline.coef_[0]:.3f}")
print(f"Baseline Model RMSE: {rmse_baseline:.2f}")
print(f"Baseline Model R²: {r2_baseline:.2f}\n")

# for each additional failure, predicted G3 decreases > failures are negative associated with final grades
# RMSE > predictions deviate by about 2.96 > in a range of 0–20, this could be enough to misidentify passing/failing
# R² is lower > shows trend too broadly and doesn't explain most variances

# ---------------------------------------------------------------------------- #
# Task 5: Build the Full Model

feature_cols = [
    "failures",
    "Medu",
    "Fedu",
    "studytime",
    "higher",
    "schoolsup",
    "internet",
    "sex",
    "freetime",
    "activities",
    "traveltime",
]
X_full = df2[feature_cols].values
y_full = df2["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

y_pred_full = model_full.predict(X_test)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
r2_full = r2_score(y_test, y_pred_full)

print(f"Train R²: {model_full.score(X_train, y_train):.3f}")
print(f"Full Model Test R²: {r2_full:.3f}")
print(f"Full Model RMSE: {rmse_full:.2f}\n")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")  # coefficients

# schoolsup > 1st largest negative coefficient > perhaps for students who have lower grades > does not reflect if it helps
# failures > 2nd largest negative coefficient > more past failures, lower predicted grade
# higher > largest positive coefficient > those wanting to pursue higher education correlates with better performance
# male students score slightly higher on average > PISA research links gap to social context, not inherent ability
# train and test R² are close > suggests model is not overfitting and generalizes well
# In production, I might keep the features with the larger coefficients that can be interpreted clearly: failures, higher, Medu, studytime, schoolsup, and sex. I might consider dropping   Fedu, activities, internet, freetime, and traveltime.

# ---------------------------------------------------------------------------- #
# Task 6: Evaluate and Summarize

# predicted vs actual plot for full model
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_full, y_test, alpha=0.7, edgecolors="k", label="Predicted vs Actual")
plt.plot(
    [0, 20],
    [0, 20],
    color="sienna",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction Line",
)
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.legend()
plt.savefig("outputs/predicted_vs_actual.png", bbox_inches="tight")
plt.close()

# Errors were roughly uniform across grade levels. A value above the diagonal means the model under-predicted that a student did better than expected, while lower mean it over-predicted the student performing worse.

print("\nSummary of findings:")
print(f"Dataset size (after): {len(df2)} students")
print(f"Test set size: {len(y_test)} students")
print(
    f"Baseline model (only 'failures') had RMSE={rmse_baseline:.2f} and R²={r2_baseline:.3f}\n"
    f"Full model (all features) had RMSE={rmse_full:.2f}, Train R²={model_full.score(X_train, y_train):.3f}, Test R²={r2_full:.3f}"
)

# largest positive coefficients > internet (+0.834) and higher (+0.610) > higher coefficient meant lower scores
# largest negative coefficients > schoolsup (-2.062) and failures: (-1.145) > lower coefficient meant lower scores
# schoolsup seems slightly surprising, but may be counterintuitive

# ---------------------------------------------------------------------------- #
# Neglected Feature: The Power of G1

feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df2[feature_cols_g1].values
y_g1 = df2["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y_g1, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

y_pred_g1 = model_g1.predict(X_test_g1)
rmse_g1 = np.sqrt(mean_squared_error(y_test_g1, y_pred_g1))
r2_g1 = r2_score(y_test_g1, y_pred_g1)

print(
    f"\nFull Model with G1:"
    f"\nTrain R²: {model_g1.score(X_train_g1, y_train_g1):.3f}"
    f"\nTest R²: {r2_g1:.3f}"
    f"\nRMSE: {rmse_g1:.2f}"
)

# Correlation is not causation; high R² here does not necessarily mean G1 is causing G3. The correlation stems from both measuring how well the student understands math. On another note, I think the model could already help to identify struggling students on G1 to decide if early action is needed. It could also help for long-term support, but I think the best model can let educators flag students via features, even before G1 any and other grade information if retrieved.