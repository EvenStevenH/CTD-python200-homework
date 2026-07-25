import requests
import pandas as pd
import sklearn
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import json
import os

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------- #
# Step 1: Fetch the Data
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 35.68,
    "longitude": 139.75,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "Asia/Tokyo",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)


print(f"Dataset loaded with {len(df)} days of data.\n")
print(df.head())
print(df.describe())
print(df.info())


# ---------------------------------------------------------------------------- #
# Step 2: Engineer Labels
def is_good_for_running(row):
    return (  # label thresholds for "good for running"
        7 <= row["temperature_2m_max"] <= 26  # 7 - 26 °C (45-79°F)
        and row["temperature_2m_min"] >= 0  # ≥ 0 °C (above freezing)
        and row["precipitation_sum"] < 3.0  # < 3.0 mm
        and row["wind_speed_10m_max"] < 30  # < 30 km/h
    )


df["good_for_running"] = df.apply(is_good_for_running, axis=1)
good_fraction = df["good_for_running"].mean()

print(
    f"Good running days: {good_fraction:.2%}\n"
    f"Class distribution:\n{df['good_for_running'].value_counts()}\n"
)

# 33.61% of days in dataset are labelled good for running. This could be reasonable, given that Tokyo lies in the humid subtropical climate zone.

# ---------------------------------------------------------------------------- #
# Step 3: Train and Tune
# split train (80%) and test (20%) sets > features and labels
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["good_for_running"], random_state=42
)
X_train = train_df.drop(columns=["good_for_running", "date"])
y_train = train_df["good_for_running"]
X_test = test_df.drop(columns=["good_for_running", "date"])
y_test = test_df["good_for_running"]

#  pipeline > Hyperparameter tuning using GridSearchCV
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))]
)
param_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)

# evaluate on the test set > test AUC
y_pred = grid_search.predict(X_test)
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)

# print > save output
print(
    f"Best C value: {grid_search.best_params_['clf__C']}\n"
    f"Best CV AUC: {grid_search.best_score_}\n"
    f"Classification Report:\n {classification_report(y_test, y_pred)}\n"
    f"Test AUC: {test_auc}\n"
)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.close()

# ---------------------------------------------------------------------------- #
# Step 4: Reflect on Evaluation

# For this dataset, the AUC score provides insight into the model's ability to distinguish between good and bad days for running (good_for_running). Its higher AUC suggests it performs well, with better overall discrimination; about what I expected, given the substantial data available (366 days).

# Precision and recall can help identify if false positives or false negatives are more prevalent, which affects usability. False negatives (instances where the model incorrectly predicts "not good for running" when it actually is) are more common than false positives.

# In practice, this means the app might slightly under-recommend running opportunities, as there are more instances classified as not good for running. I prefer this over a higher count of false positives; identifying more non-running days over minimizing missed opportunities seems like a fair trade-off.

# For setting the threshold, the default 0.5 might be too lenient given that we have more "False" (not good for running) cases. A higher threshold would reduce false positives but increase false negatives. Given the slightly better recall for the positive class, I would consider using a threshold around 0.6–0.7 if I wanted to be more conservative when recommending a run. This would reduce false positives by only recommending running when the model is more confident, although it would also increase false negatives by missing some acceptable running days.


# ---------------------------------------------------------------------------- #
# Step 5: Save the Model (best pipeline)
joblib.dump(grid_search.best_estimator_, "models/weather_classifier.pkl")

metadata = {
    "python_version": sys.version,
    "scikit-learn_version": sklearn.__version__,
    "feature_names": list(X_train.columns),
    "best_params": grid_search.best_params_,
    "test_auc": test_auc,
    "city_latitude": params["latitude"],
    "city_longitude": params["longitude"],
    "label_thresholds_description": (
        "Label 'good for running' if temperature_2m_max is 7-26°C, "
        "temperature_2m_min >= 0°C, precipitation_sum < 3.0 mm, and wind_speed_10m_max < 30 km/h."
    ),
    "trained_on": "2024 Open-Meteo, Japan (lat 35.68, lon 139.75)",
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Model saved successfully.")
