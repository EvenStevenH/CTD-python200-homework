import os
import glob
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

# Pre-task: Load the Data
RAW_DATA_DIR = "../assignments_01/inputs"
DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"
OUTPUTS_DIR = "./outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
df = None  # global DataFrame to hold loaded data

# ---------------------------------------------------------------------------- #
# Task 1: Define Your Tools


def merge_df() -> pd.DataFrame:
    """Load and merge all yearly CSVs into a single cleaned DataFrame."""
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    yearly_frames = []
    for path in sorted(csv_files):
        frame = pd.read_csv(path, sep=";", decimal=",")
        basename = os.path.basename(path).replace(".csv", "")
        frame["year"] = int(basename[-4:])
        frame.columns = frame.columns.str.strip()
        if "Ladder score" in frame.columns:
            frame.rename(columns={"Ladder score": "Happiness score"}, inplace=True)
        yearly_frames.append(frame)
    merged = pd.concat(yearly_frames, ignore_index=True)
    merged.columns = [c.lower().strip().replace(" ", "_") for c in merged.columns]
    return merged


@tool  # tool 1
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory by merging all yearly CSVs.

    Loads and concatenates all CSV files found in the happiness_project directory.
    Each file represents one year of data.

    Returns:
        A plain dict with keys 'num_rows' (int), 'num_columns' (int), and
        'column_names' (list of str), or an 'error' key. This is a dict, NOT a
        DataFrame - read values with e.g. result['num_rows'], never result.shape
        or result.columns.
    """
    global df
    if df is not None:
        print("Data already loaded. Returning existing dataframe info.")
        return {
            "num_rows": df.shape[0],
            "num_columns": df.shape[1],
            "column_names": list(df.columns),
        }

    if os.path.exists(DATA_PATH):
        print(f"Loading merged file from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("Pre-merged file not found. Merging yearly CSVs.")
        df = merge_df()

    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": list(df.columns),
    }


@tool  # tool 2
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The snake_case column name to summarize (e.g. 'happiness_score').

    Returns:
        A dict with the result of describe() or an "error" key if something goes wrong.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}
    return df[column].describe().to_dict()


@tool  # tool 3
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Uses scipy.stats.pearsonr. Rows with NaN in either column are dropped before computing.

    Args:
        col1: Name of the first column.
        col2: Name of the second column.

    Returns:
        A dict with keys 'col1', 'col2', 'pearson_r', and 'p_value' (rounded to 4 decimal
        places), or an 'error' key on bad input.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"One of the columns ({col1}, {col2}) does not exist."}
    try:
        cols = df[[col1, col2]].dropna()
        r, p = stats.pearsonr(cols[col1], cols[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4),
        }
    except Exception as e:
        return {"error": str(e)}


@tool  # tool 4
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Filters the dataset to the given year, sorts by the column descending, and returns
    the top N rows.

    Args:
        column: The column to rank countries by (snake_case, must be numeric).
        year: The year to filter on (e.g., 2019, 2020).
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A dict with key 'top_countries' containing a list of dicts, each with 'country' and
        the requested column value. Returns an 'error' key on bad input.
    """
    global df
    if df is None:
        return {"error": "No data loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if "year" not in df.columns:
        return {"error": "No 'year' column found in the dataset."}

    year_df = df[df["year"] == year]  # filter by year, sort descending
    top = year_df.sort_values(column, ascending=False).head(n)
    results = [
        {"country": row["country"], column: row[column]} for _, row in top.iterrows()
    ]
    return {"top_countries": results}


# ---------------------------------------------------------------------------- #
# Task 2: Build the Agent > instantiate a CodeAgent

model = OpenAIServerModel(api_key=os.environ["OPENAI_API_KEY"], model_id="gpt-4o-mini")
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
When you do write custom code, do NOT try to reconstruct the dataset from a tool's
return value (those only contain summaries/metadata, not row-level data). Instead, once
load_happiness_data has been called, the full dataset is available directly in your
Python environment as the variable `happiness_df` (a pandas DataFrame) - use it directly.
Be concise and student-friendly in your responses.
"""
agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries,
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# ---------------------------------------------------------------------------- #


def _run_query(query: str):
    """Run a query, exposing the current DataFrame (if loaded) to the agent's
    Python environment as `happiness_df` so custom code can use the real data
    instead of trying to rebuild it from a tool's summary/metadata output."""
    global df
    extra_args = {"happiness_df": df} if df is not None else None
    return agent.run(query, reset=False, additional_args=extra_args)


if __name__ == "__main__":
    # Task 3: Run Guided Queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per regional_indicator. Save the plot to outputs/happiness_by_region.png.",
    ]
    for query in queries:
        response = _run_query(query)
        print(response)

    plot_path = "outputs/happiness_by_region.png"
    if os.path.exists(plot_path):
        print(f"Plot confirmed saved at: {plot_path}")
    else:
        print(f"Plot not found at {plot_path}")

    # Task 4: Your Own Questions
    my_query_1 = "What are the mean happiness scores by region?"
    response_1 = _run_query(my_query_1)
    print(f"My Query 1: {my_query_1}")
    print(response_1)
    # This triggered code generation, since none of the tools computes mean happiness scores by region.

    my_query_2 = """
        Plot healthy_life_expectancy vs. happiness_score for the year 2020.
        Label axes and give the chart a title: 'Life Expectancy vs Happiness (2020)'.
        Save it as outputs/expectancy_vs_happiness.png.
        """
    response_2 = _run_query(my_query_2)
    print(f"My Query 2: {my_query_2}")
    print(response_2)
    # This triggered code generation, since none of the four tools produce plots. The agent writes its own matplotlib code to create and save the scatter plot.

# ---------------------------------------------------------------------------- #
# --- Task 5: Reflection

# 1) The agent used compute_correlation to get pearson_r=0.6313, p_value=0.0, and 'statistical_significance'=True. The agent implicitly uses the standard 0.05 threshold with the p_value of 0.0 being below it, so the classification is correct. The pearson_r of 0.63 indicates a moderate positive correlation; higher GDP per capita is associated with higher happiness scores.

# 2) Query 5 surprised me by being less capable than I expected; it was able to pass the correct arguments and create a scatter plot, but it did not adjust it to be human-readable. That us, it created a legend for each line color, but the legend is cropped by the edges of the image and doesn't allow one to see what each color stands for.

# 3) An additional tool that could make this agent meaningfully more useful is a dedicated function for comparing countries. It would take a list of country names and a column name, and then return each country's value for every year as a result. It could answer a question like "Compare happiness_score between the United States and Africa" and return a predictable, structured result without needing the agent to write code that performs extra steps (like merging or filtering data, which increases the potential for errors through typos or logical mistakes).