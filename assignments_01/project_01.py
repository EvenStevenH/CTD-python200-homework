import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import task, flow, get_run_logger  # prefect server start

# Part 2: Mini-Project: World Happiness Pipeline
matplotlib.use("Agg")  # non-interactive backend > save plots w/o display

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
YEARS = list(range(2015, 2025))
COL_ALIASES = {"ladder_score": "happiness_score"}


# ---------------------------------------------------------------------------- #
@task(retries=3, retry_delay_seconds=2)
def load_data(years, data_dir, output_dir):
    logger = get_run_logger()
    logger.info("------ Task 1: Load Multiple Years of Data")

    dfs = []
    for year in years:
        path = os.path.join(data_dir, f"world_happiness_{year}.csv")
        if not os.path.exists(path):
            logger.warning(f"Skipping missing file: {path}")
            continue
        try:
            df = pd.read_csv(path, sep=";", decimal=",")
        except Exception as e:
            logger.warning(f"Skipping unreadable file: {path}")
            logger.warning(f"Error: {e}")
            continue

        # standardize col names > rename aliases to canonical > add year col
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        df.rename(
            columns={key: val for key, val in COL_ALIASES.items() if key in df.columns},
            inplace=True,
        )
        df["year"] = year
        dfs.append(df)
        logger.info(f"Loaded data for {year} ({len(df)} rows)")
    if not dfs:
        raise ValueError(
            f"Unable to load from {data_dir}. Folder might not exist or contain CSV files."
        )

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.loc[:, ~merged.columns.duplicated()]  # drop dupe cols
    out_path = os.path.join(output_dir, "merged_happiness.csv")
    merged.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(merged)} rows)")
    return merged


# ---------------------------------------------------------------------------- #
@task
def descriptive_stats(df):
    logger = get_run_logger()
    logger.info("------ Task 2: Descriptive Statistics for happiness_score")

    logger.info(f"-- OVERALL STATS")
    logger.info(f"mean: {df["happiness_score"].mean():.4f}")
    logger.info(f"median: {df["happiness_score"].median():.4f}")
    logger.info(f"std: {df["happiness_score"].std():.4f}")

    logger.info("-- MEAN by YEAR")
    for year, mean in df.groupby("year")["happiness_score"].mean().items():
        logger.info(f"{year}: {mean:.4f}")

    logger.info("-- MEAN by REGION")
    for region, mean in (
        df.groupby("regional_indicator")["happiness_score"]
        .mean()
        .sort_values(ascending=False)
    ).items():
        logger.info(f"{region}: {mean:.4f}")

    return df


# ---------------------------------------------------------------------------- #
@task
def visual_exploration(df, output_dir):
    logger = get_run_logger()
    logger.info("------ Task 3: Visual Exploration")

    # histogram > all happiness scores across all years
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["happiness_score"].dropna(), bins=40, edgecolor="sienna", color="tan")
    ax.set_title("Distribution of Happiness Scores (2015–2024)")
    ax.set_xlabel("Happiness Score")
    ax.set_ylabel("Count")
    out_path = os.path.join(output_dir, "happiness_histogram.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")

    # boxplot > happiness score distributions across years
    fig, ax = plt.subplots(figsize=(10, 6))
    years_sorted = sorted(df["year"].unique())
    data_by_year = [
        df[df["year"] == y]["happiness_score"].dropna().values for y in years_sorted
    ]
    ax.boxplot(data_by_year, tick_labels=years_sorted)
    ax.set_title("Happiness Score Distribution by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Happiness Score")
    out_path = os.path.join(output_dir, "happiness_by_year.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")

    # scatter plot > GDP per capita vs happiness score
    gdp_col = "gdp_per_capita"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        df[gdp_col].dropna(),
        df["happiness_score"].dropna(),
        alpha=0.4,
        s=10,
        color="sienna",
    )
    ax.set_title("GDP per Capita vs Happiness Score")
    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Happiness Score")
    out_path = os.path.join(output_dir, "gdp_vs_happiness.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")

    # heatmap > Pearson correlations between all numeric columns
    numeric_cols = df.select_dtypes(include="number").drop(
        columns=["ranking", "year"], errors="ignore"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="BrBG", ax=ax)
    ax.set_title("Pearson Correlation Heatmap")
    out_path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ---------------------------------------------------------------------------- #
@task
def hypothesis_testing(df):
    logger = get_run_logger()
    logger.info("------ Task 4: Hypothesis Testing")

    # T-test > 2019 vs 2020 happiness scores
    a = df[df["year"] == 2019]["happiness_score"].dropna()
    b = df[df["year"] == 2020]["happiness_score"].dropna()
    t_stat1, p_val1 = stats.ttest_ind(a, b)
    logger.info(f"-- T-test: 2019 vs 2020 happiness scores")
    logger.info(f"2019 mean: {a.mean():.4f}")
    logger.info(f"2020 mean: {b.mean():.4f}")
    logger.info(f"t-statistic: {t_stat1:.4f}")
    logger.info(f"p-value: {p_val1:.4f}")

    if p_val1 < 0.05:
        ttest1_conclusion = f"There IS a statistically significant difference (p < 0.05). Global happiness was notably {"lower" if b.mean() < a.mean() else "higher"} in 2020 than in 2019, suggesting the pandemic was associated with a measurable change."
    else:
        ttest1_conclusion = "NO statistically significant difference (p ≥ 0.05). We cannot conclude that the pandemic affected global happiness scores."
    logger.info(f"Conclusion: {ttest1_conclusion}")

    # T-test > Western Europe vs South Asia
    c = df[df["regional_indicator"] == "Western Europe"]["happiness_score"].dropna()
    d = df[df["regional_indicator"] == "South Asia"]["happiness_score"].dropna()
    t_stat2, p_val2 = stats.ttest_ind(c, d)
    logger.info("-- T-test: Western Europe vs South Asia")
    logger.info(f"Western Europe mean: {c.mean():.4f}")
    logger.info(f"South Asia mean: {d.mean():.4f}")
    logger.info(f"t-statistic: {t_stat2:.4f}")
    logger.info(f"p-value: {p_val2:.4f}")
    if p_val2 < 0.05:
        logger.info(
            f"Conclusion: there IS a statistically significant difference (p < 0.05). Western Europe reports notably {"lower" if c.mean() < d.mean() else "higher"} happiness scores than South Asia across all years."
        )
    else:
        logger.info(
            "Conclusion: NO statistically significant difference (p ≥ 0.05) in happiness scores between Western Europe and South Asia."
        )

    return ttest1_conclusion


@task
def correlation_analysis(df):
    logger = get_run_logger()
    logger.info("------ Task 5: Correlation and Multiple Comparisons")

    numeric_cols = df.select_dtypes(include="number").drop(
        columns=["ranking", "year", "happiness_score"], errors="ignore"
    )
    results = {}
    for var in numeric_cols.columns.tolist():  # explanatory variables
        subset = df[[var, "happiness_score"]].dropna()
        r, p = stats.pearsonr(subset[var], subset["happiness_score"])

        results[var] = {  # store in structured format
            "r": float(r),  # coefficient
            "p": float(p),  # p_value
            "sig_og": bool(p < 0.05),  # unadjusted
            "sig_adj": False,  # adjusted (placeholder)
        }

    n_tests = len(results)  # number of variables analyzed
    adjusted_alpha = 0.05 / n_tests  # calculate Bonferroni correction
    for var, result in results.items():
        result["sig_adj"] = bool(result["p"] < adjusted_alpha)  # determine significance
        logger.info(
            f"{var}: r={result['r']:.3f} p={result['p']:.4g} sig_og={result['sig_og']} sig_adj={result['sig_adj']}"
        )

    strongest_var = max(results, key=lambda var: abs(results[var]["r"]))

    logger.info(f"Number of variables analyzed: {n_tests}")
    logger.info(f"Adjusted_alpha: {adjusted_alpha:.4g}")
    return {"strongest_var": strongest_var}


@task
def summary_report(df, hypothesis, cor_results):
    logger = get_run_logger()
    logger.info("------ Task 6: Summary Report")

    logger.info(f"Total number of countries: {df["regional_indicator"].nunique()}")
    logger.info(f"Total number of years: {df["year"].nunique()}")

    logger.info("-- Top 3 regions by mean happiness score:")
    for region, score in (
        df.groupby("regional_indicator")["happiness_score"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    ).items():
        logger.info(f"{region}: {score:.4f}")

    logger.info("-- Bottom 3 regions by mean happiness score:")
    for region, score in (
        df.groupby("regional_indicator")["happiness_score"]
        .mean()
        .sort_values(ascending=True)
        .head(3)
    ).items():
        logger.info(f"{region}: {score:.4f}")

    logger.info(f"Conclusion for pre/post-2020 t-test: {hypothesis}")
    logger.info(
        f"Variable most strongly correlated with happiness score (after Bonferroni correction): {cor_results["strongest_var"]}"
    )


@flow
def happiness_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(YEARS, DATA_DIR, OUTPUT_DIR)
    df = descriptive_stats(df)
    visual_exploration(df, OUTPUT_DIR)
    summary_report(df, hypothesis_testing(df), correlation_analysis(df))


if __name__ == "__main__":
    happiness_pipeline()
