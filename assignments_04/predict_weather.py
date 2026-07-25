import json
import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------- #
# Task 1: Load and Verify
model_path = Path("models/weather_classifier.pkl")
metadata_path = Path("models/weather_classifier_metadata.json")
if not model_path.exists() or not metadata_path.exists():
    raise FileNotFoundError(
        "Model files not found. Run train_weather_classifier.py first."
    )

model = joblib.load(model_path)
with open(metadata_path, "r") as f:
    metadata = json.load(f)

print(
    "Model Metadata\n"
    f"City: ({metadata['city_latitude']}, {metadata['city_longitude']})\n"
    f"Features: {metadata['feature_names']}\n"
    f"Test AUC: {metadata['test_auc']:.3f}\n"
)

# ---------------------------------------------------------------------------- #
# Task 2: Predict on New Data

feature_names = metadata["feature_names"]

new_days = pd.DataFrame(
    [
        [22, 12, 0.0, 12],  # good
        [35, 25, 0.0, 10],  # slightly warmer
        [15, 7, 15.0, 18],  # rainy
        [18, 8, 1.0, 45],  # windy
        [26, 0, 2.9, 29],  # borderline
    ],
    columns=feature_names,
)

predictions = model.predict(new_days)
probabilities = model.predict_proba(new_days)[:, 1]

print("Predictions")
for i, row in new_days.iterrows():
    label = "Good" if predictions[i] else "Skip"
    print(
        f"Day {i+1}\n"
        f"{row.to_dict()}\n"
        f"Prediction: {label}\n"
        f"Confidence: {probabilities[i]:.2%}\n"
    )

# ---------------------------------------------------------------------------- #
# Task 3: Reflection

# The borderline case (26°C high, 0°C low, 2.9 mm precipitation,and 29 km/h wind) produced a probability (3.67%) close to the decision boundary.

# A probability around 0.52 indicates the model is fairly uncertain, since it is only slightly above the default threshold. In a real application, I might use a higher threshold if I wanted more conservative recommendations.

# If someone runs predict_weather.py before train_weather_classifier.py, the saved model and metadata files will not exist, causing the program to fail. The FileNotFoundError at the beginning provides a clearer, helpful error message telling the user to run the training script first.

# In production, this script would likely retrieve tomorrow's weather forecast from a weather API instead of using manually created sample data. After downloading the forecast, it would build a DataFrame with the same feature names, load the saved model, and generate predictions automatically each day.
