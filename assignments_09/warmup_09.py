import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import date

# ---------------------------------------------------------------------------- #
# --- Supabase Connection ---

# Connection Question 1
# The two pieces of information supabase-py needs to connect to your project are SUPABASE_URL (found in the dashboard under "Project Settings" > "Project URL") and SUPABASE_KEY (found in the dashboard under "API"). These should never be hardcoded in a Python script because they are sensitive credentials that could be used maliciously if exposed. They should always be stored in environment variables (.env file) and loaded using python-dotenv.


# ---------------------------------------------------------------------------- #
# Connection Question 2
def get_client():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError("SUPABASE_URL environment variable not set")
    if not key:
        raise ValueError("SUPABASE_KEY environment variable not set")
    return create_client(url, key)


supabase = get_client()

# ---------------------------------------------------------------------------- #
# Connection Question 3
# Row Level Security (RLS) is a feature that allows you to define fine-grained access policies on tables, letting you specify who can read or write data based on roles and conditions. In this course, we disabled RLS because it adds complexity during development; our focus is learning to build a functional pipeline first. RLS should be enabled in production applications where you want to prevent unauthorized access to sensitive data, maintain data integrity and consistency across teams, and meet requirements in compliance frameworks (like HIPAA).


# ---------------------------------------------------------------------------- #
# --- Supabase Connection ---
# CRUD Question 1
def insert_test_record(supabase):
    test_record = {
        "date": "2023-06-15",
        "temperature_2m_max": 28.4,
        "temperature_2m_min": 17.2,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.1,
    }
    return supabase.table("weather_raw").insert(test_record).execute()


# insert_test_record(supabase)
# Running the function twice would result in an error because we're trying to insert a record with the same primary key (date). To make it safe for multiple runs, we should use upsert() instead of insert().


# ---------------------------------------------------------------------------- #
# CRUD Question 2
def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", str(start))
        .lte("date", str(end))
        .execute()
    )
    return response.data


records = get_records_by_date_range(supabase, "2022-01-01", "2026-12-31")
print(f"Found {len(records)} record(s):")
for r in records:
    print(r, "\n")

# ---------------------------------------------------------------------------- #
# CRUD Question 3
# In Supabase, insert() is used to add new records into a table, while upsert() is a combination of an INSERT and UPDATE operation. An upsert() will insert a record if the specified condition (key) is not already present in the table. If the key exists then it will update the existing row instead of inserting a new one. An example scenario; in an e-commerce app where I need to add new products to an inventory table, I'd use insert() to ensure that adding duplicate products does not inadvertently update existing records incorrectly or without notice, and then upsert() to update prices of existing products.


def safe_upsert(supabase, records):
    response = supabase.table("weather_raw").upsert(records).execute()
    print(f"Number of rows affected: {len(response.data)}")


test_records = [
    {
        "date": "2023-06-16",
        "temperature_2m_max": 28.4,
        "temperature_2m_min": 17.2,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12.1,
    },
    {
        "date": "2023-06-17",
        "temperature_2m_max": 26.4,
        "temperature_2m_min": 15.5,
        "precipitation_sum": 0.2,
        "wind_speed_10m_max": 11.1,
    },
]
safe_upsert(supabase, test_records)

# ---------------------------------------------------------------------------- #
# --- Idempotency ---
# Idempotency Question 1

# Idempotency means that running an operation multiple times produces the same result as running it once. In a data pipeline, this matters because pipelines are often retried (due to network failures, worker crashes, or manual re-runs) and you don't want those retries to corrupt your data. In a non-idempotent pipeline when the script crashes halfway through and is restarted (in, let's say, a pipeline that reads lines from a log file containing user actions and writes each line's timestamped events into a database), it might cause it to reprocess previously read entries and lead to duplicate records.
