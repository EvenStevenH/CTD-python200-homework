from prefect import task, flow
import pandas as pd
import numpy as np

# Pipeline Question 2
arr = np.array(
    [12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0]
)


@task
def create_series(arr):
    return pd.Series(arr, name="values")


@task
def clean_data(series):
    return series.dropna()


@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }


@flow
def pipeline_flow(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary


if __name__ == "__main__":
    result = pipeline_flow(arr)
    for key, value in result.items():
        print(f"{key}: {value}")

# I think Prefect is designed for more complex workflows and may be a bit excessive for a straightforward data processing task — three small functions on a handful of numbers. The overhead of setting up tasks/flows also outweighs any significant benefits. On the other hand, some realistic scenarios where a framework like Prefect could still be useful are those needing monitoring tools to track the status of pipeline runs, scalability as data processing needs grow, and orchestration/scheduling.
