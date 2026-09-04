import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# https://youtu.be/sokVZceUyig

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------- #
# Step 1: Incremental Read
with open("./models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
feature_names = metadata.get("feature_names", [])

raw_rows = supabase.table("weather_raw").select("*").execute().data or []
enriched_dates = {
    r["date"]
    for r in (supabase.table("weather_enriched").select("date").execute().data or [])
}
to_classify = [r for r in raw_rows if r.get("date") not in enriched_dates]

print(f"Total raw records: {len(raw_rows)}")
print(f"Already enriched: {len(enriched_dates)}")
print(f"Records to process this run: {len(to_classify)}")
if not to_classify:
    print("Nothing to do — all records already enriched.")
    exit()

# ---------------------------------------------------------------------------- #
# Step 2: ML Transform
clf = joblib.load("./models/weather_classifier.pkl")
df = pd.DataFrame(to_classify)
X = df[feature_names]

predictions = clf.predict(X)
probabilities = clf.predict_proba(X)[:, 1]

good_count = sum(1 for p in predictions if bool(p))
conf_min = round(float(min(probabilities)), 4)
conf_max = round(float(max(probabilities)), 4)

print(f"ML Predictions Summary: {good_count} days classified as 'good for running'")
print(f"Confidence range: {conf_min:.0%} to {conf_max:.0%}")

enrichment_records = [
    {
        "date": to_classify[i]["date"],
        "good_for_running": bool(predictions[i]),
        "confidence": round(float(probabilities[i]), 4),
        "llm_summary": None,
    }
    for i in range(len(to_classify))
]

# ---------------------------------------------------------------------------- #
# Step 3: LLM Transform
SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)


def make_user_message(row, good_for_running, confidence):
    prediction_text = (
        "good for running" if good_for_running else "not ideal for running"
    )
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )


for i, record in enumerate(enrichment_records):
    raw_row = to_classify[i]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": make_user_message(
                        raw_row, record["good_for_running"], record["confidence"]
                    ),
                },
            ],
            max_tokens=100,
        )
        summary = response.choices[0].message.content.strip()
        record["llm_summary"] = summary or "Recommendation unavailable."
    except Exception as e:
        print(f"API error on {record['date']}: {e}")
        record["llm_summary"] = "Recommendation unavailable."

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1} / {len(enrichment_records)}")

# ---------------------------------------------------------------------------- #
# Step 4: Load
db_response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)
upserted_count = len(db_response.data) if db_response.data else len(enrichment_records)
print(f"Upserted {upserted_count} rows into weather_enriched")

# ---------------------------------------------------------------------------- #
# Step 5: Verify
all_enriched = supabase.table("weather_enriched").select("*").execute().data or []
total_rows = len(all_enriched)
good_days = sum(1 for r in all_enriched if r.get("good_for_running"))

print(f"Total rows in weather_enriched: {total_rows}")
print(f"Days classified as good for running (all time): {good_days}")
print("Sample rows:")
for row in all_enriched[:5]:
    print(
        f"Date: {row['date']} | Good: {row.get('good_for_running')} | Confidence: {row.get('confidence'):.2f} | Summary: {row.get('llm_summary', 'N/A')}"
    )

# Looking at the LLM summaries, they seem to consistently align with both the raw weather features and the model's binary prediction. A particularly good one was on 2023-01-10, where the summary was, "Today's cool temperatures and clear skies with light winds make it an excellent day for running", with a confidence level of 0.9656. One that seemed off was on 2023-09-16, where the summary was, "Despite mild temperatures and no rain, the wind and low confidence in the prediction suggest it may not be the best day for running", with a confidence level of 0.4656. This weaker one might have been caused by the LLM choosing strict threshold adherence, despite describing a day that would be fit for running.

# ---------------------------------------------------------------------------- #
# Step 6: Reflect
# 1) The classifier should remain accurate for other cities (besides Charlotte, NC) because comfort thresholds for running are largely universal, and the model relies on standardized features rather than location-specific patterns. That is, even if local climate differ slightly, the logistic regression boundaries for temperature, precipitation, and wind translate well to most areas without requiring retraining. 2) The LLM is a purely additive layer that translates structured predictions into natural language without overriding the classifier; this preserves deterministic logic when generating recommendations, ensuring the original data is untouched. 3) My main concerns for scaling to 50,000 records would be API cost and rate-limit latency. I might address this by batching identical weather states, implementing a function for retrying a failed operation, and caching results to eliminate redundant calls.
