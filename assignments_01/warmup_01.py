import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
import warnings

# # Part 1: Warmup Exercises
warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------------------------- #
# --- Pandas Review ---

# Pandas Question 1
data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True],
}
df = pd.DataFrame(data)
print(f"Num Rows: {len(df)}\n")
print(f"First 3 Rows: {df.head(3)}\n")
print(f"Shape: {df.shape}\n")
print(f"Data Types: {df.dtypes}\n")

# Pandas Question 2
print("Students who passed and have a grade above 80:")
print(df[(df["passed"] == True) & (df["grade"] > 80)], "\n")

# Pandas Question 3
df["grade_curved"] = df["grade"] + 5
print("New col > add 5 points to each grade:")
print(df, "\n")

# Pandas Question 4
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]], "\n")

# Pandas Question 5
mean_grades = df.groupby("city")["grade"].mean()
print(mean_grades, "\n")

# Pandas Question 6
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]], "\n")

# Pandas Question 7
sorted_df = df.sort_values(by="grade", ascending=False)
print("Top Three Grades:")
print(sorted_df.head(3), "\n")

# ---------------------------------------------------------------------------- #
# --- NumPy Review ---

# NumPy Question 1
arr_q1 = np.array([10, 20, 30, 40, 50])
print("1D array:\n", arr_q1)
print(f"Shape: {arr_q1.shape}")
print(f"Data Types: {arr_q1.dtype}")
print(f"Number of array dimensions: {arr_q1.ndim}\n")

# NumPy Question 2
arr_q2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D array:\n", arr_q2)
print(f"Shape: {arr_q2.shape}")
print(f"Size: {arr_q2.size}\n")

# NumPy Question 3
arr_q3 = arr_q2[:2, :2]  # first two rows and columns
print("Top-left 2x2 block:\n", arr_q3, "\n")

# NumPy Question 4
arr_q4_zeros = np.zeros((3, 4))
arr_q4_ones = np.ones((2, 5))
print(f"Array of zeroes:\n {arr_q4_zeros}")
print(f"Array of ones:\n {arr_q4_ones}\n")

# NumPy Question 5
arr_q5 = np.arange(0, 50, 5)
print(f"Array:\n {arr_q5}")
print(f"Shape: {arr_q5.shape}")
print(f"Mean: {arr_q5.mean()}")
print(f"Sum: {arr_q5.sum()}")
print(f"Standard Deviation: {arr_q5.std()}\n")

# NumPy Question 6
arr_q6 = np.random.normal(0, 1, 200)  # mean 0 > standard deviation 1
print(f"Random Values Mean: {arr_q6.mean()}")
print(f"Random Values Standard Deviation: {arr_q6.std()}\n")

# ---------------------------------------------------------------------------- #
# --- Matplotlib Review ---

# Matplotlib Question 1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Question 2
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Scores")
plt.show()

# Matplotlib Question 3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Matplotlib Question 4
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(x, y)  # Q1 as line
axs[0].set_title("Squares")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[1].bar(subjects, scores)  # Q2 as bar plot
axs[1].set_title("Subject Scores")
axs[1].set_xlabel("Subjects")
axs[1].set_ylabel("Scores")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------- #
# --- Descriptive Statistics Review ---

# Descriptive Stats Question 1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
data_q1 = np.array(data)
print(f"Statistics Q1:\n {data_q1}")
print(f"Mean: {data_q1.mean()}")
print(f"Median: {np.median(data_q1)}")
print(f"Variance: {data_q1.var()}")
print(f"Standard Deviation: {data_q1.std()}\n")

# Descriptive Stats Question 2
random_scores = np.random.normal(65, 10, 500)  # mean 65 > standard deviation 10
plt.hist(random_scores, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Question 3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()

# Descriptive Stats Question 4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()

# The exponential distribution is more skewed in comparison, with the median descriptive statistic likely providing a more appropriate measure of central tendency because it's less affected by extreme values and outliers. For the normal distribution, both mean and median might be okay (as the distribution is nearly symmetric), but I might go with mean for its mathematical properties.

# Descriptive Stats Question 5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
print(f"Data1 Mean: {np.mean(data1)}")
print(f"Data1 Median: {np.median(data1)}")
print(f"Data1 Mode: {stats.mode(data1).mode[0]}")
print(f"Data2 Mean: {np.mean(data2)}")
print(f"Data2 Median: {np.median(data2)}")
print(f"Data2 Mode: {stats.mode(data2).mode[0]}")

# I think the median and mean are so different for data2 because of the outlier (150), which skews the distribution's mean upwards while the median remains relatively unchanged.

# ---------------------------------------------------------------------------- #
# --- Hypothesis Testing Review ---

# Hypothesis Question 1 > independent samples t-test
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat1, p_val1 = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat1}")
print(f"P-value: {p_val1}\n")

# Hypothesis Question 2 > alpha = 0.05
if p_val1 < 0.05:
    print("P-value result is statistically significant.\n")
else:
    print("P-value result is not statistically significant.\n")

# Hypothesis Question 3 > paired t-test
before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]
t_stat3, p_val3 = stats.ttest_rel(before, after)
print(f"T-statistic: {t_stat3}")
print(f"P-value: {p_val3}\n")

# Hypothesis Question 4 > one-sample t-test
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val = stats.ttest_1samp(scores, 70)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}\n")

# Hypothesis Question 5 > one-tailed test
t_stat4, p_val4 = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"T-statistic: {t_stat4}")
print(f"P-value: {p_val4}\n")

# Hypothesis Question 6
if p_val1 < 0.05:
    print(
        "Statistically significant difference between scores, with group_a having lower scores than group_b. Unlikely due to chance.\n"
    )
else:
    print(
        "No statistically significant difference between group_a and group_b scores. Suggests that any observed difference could be due to chance.\n"
    )

# ---------------------------------------------------------------------------- #
# --- Correlation Review ---

# Correlation Question 1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_q1 = np.corrcoef(x, y)  # Pearson correlation
print(f"Correlation Matrix:\n{corr_q1}")
print(f"Correlation Coefficient: {corr_q1[0, 1]}\n")  # correlation coefficient
# Expected correlation to be 1, because y is a perfect positive linear transformation of x; y = 2x.

# Correlation Question 2
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]
corr_coef, p_val = pearsonr(x, y)
print(f"Correlation Coefficient: {corr_coef}")
print(f"P-value: {p_val}\n")

# Correlation Question 3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28],
}
df = pd.DataFrame(people)
corr_q3 = df.corr()
print(f"Correlation Matrix:\n{corr_q3}\n")

# Correlation Question 4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Correlation Question 5
sns.heatmap(corr_q3, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------------------------------- #
# Pipelines

# Pipeline Question 1
arr = np.array(
    [12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0]
)  # contains some missing values


def create_series(arr):
    return pd.Series(arr, name="values")


def clean_data(series):
    return series.dropna()


def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],  # get single value
    }


def data_pipeline(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary


result = data_pipeline(arr)
for key, value in result.items():
    print(f"{key}: {value}")
