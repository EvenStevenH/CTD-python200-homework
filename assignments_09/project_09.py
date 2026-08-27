import os
import requests
from dotenv import load_dotenv
from datetime import date
from supabase import create_client

# https://youtu.be/WAZ-NS0Bwik


# ---------------------------------------------------------------------------- #
def get_client():  # Set up Supabase
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError("SUPABASE_URL environment variable not set")
    if not key:
        raise ValueError("SUPABASE_KEY environment variable not set")
    return create_client(url, key)


# ---------------------------------------------------------------------------- #
def get_weather_data(city):  # Step 1: Extract
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={city['latitude']}&longitude={city['longitude']}&"
        f"start_date=2023-01-01&end_date=2023-12-31&"
        "daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
    )
    response.raise_for_status()
    weather_data = response.json()
    print(f"Data for {city['name']} retrieved from Open-Meteo API!")
    print(f"Record Count: {len(response.json()['daily']['time'])}")
    print(f"Keys: {weather_data.keys()}\n")
    return weather_data


# ---------------------------------------------------------------------------- #
def transform(data):  # Step 2: Transform
    daily = data["daily"]
    dates = daily["time"]

    records = []  # columnar format into a list of row dicts
    for i in range(len(dates)):
        record = {
            "date": dates[i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }
        records.append(record)

    print(f"First record: {records[0]}")
    print(f"Last record: {records[-1]}")
    print(f"Expected 365 records. Retrieved: {len(records)} records\n")
    return records


# ---------------------------------------------------------------------------- #
def load(records):  # Step 3: Load
    result = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    print(f"Loaded {len(records)} rows into weather_raw")
    return result


# ---------------------------------------------------------------------------- #
def verify():  # Step 4: Verify
    total = supabase.table("weather_raw").select("*", count="exact").execute()
    earliest = (
        supabase.table("weather_raw")
        .select("date")
        .order("date", desc=False)
        .limit(1)
        .execute()
    )
    latest = (
        supabase.table("weather_raw")
        .select("date")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    july_4 = (
        supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()
    )
    print(f"Total rows in weather_raw: {total.count}")
    print(f"Earliest date: {earliest.data[0]['date']}")
    print(f"Latest date: {latest.data[0]['date']}\n")
    print(f"2023-07-04 data:")
    for key, val in july_4.data[0].items():
        print(f"{key}: {val}")


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":  # Run pipeline
    supabase = get_client()
    city = {"name": "New York", "latitude": 40.7128, "longitude": -74.0060}
    data = get_weather_data(city)
    records = transform(data)
    load(records)
    verify()
    # After transforming the data, I expected 365 records for a full year and got 365 records as a result.
    # I ran the script a second time and confirmed that the row count in weather_raw does not change. This tells me that the script is idempotent; duplicate records will not be made.
