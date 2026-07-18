import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# machine learning modules
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    accuracy_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------- #
print("\n\n------ Task 1: Load and Explore")

COLUMN_NAMES = [
    "word_freq_make",  # 0   percent of words that are "make"
    "word_freq_address",  # 1
    "word_freq_all",  # 2
    "word_freq_3d",  # 3   almost never appears
    "word_freq_our",  # 4
    "word_freq_over",  # 5
    "word_freq_remove",  # 6   common in "remove me from this list"
    "word_freq_internet",  # 7
    "word_freq_order",  # 8
    "word_freq_mail",  # 9
    "word_freq_receive",  # 10
    "word_freq_will",  # 11
    "word_freq_people",  # 12
    "word_freq_report",  # 13
    "word_freq_addresses",  # 14
    "word_freq_free",  # 15  classic spam word
    "word_freq_business",  # 16
    "word_freq_email",  # 17
    "word_freq_you",  # 18
    "word_freq_credit",  # 19
    "word_freq_your",  # 20  often high in spam
    "word_freq_font",  # 21  HTML emails
    "word_freq_000",  # 22  "win $ x,000" style offers
    "word_freq_money",  # 23  money related
    "word_freq_hp",  # 24  HP specific
    "word_freq_hpl",  # 25
    "word_freq_george",  # 26  specific HP person
    "word_freq_650",  # 27  area code
    "word_freq_lab",  # 28
    "word_freq_labs",  # 29
    "word_freq_telnet",  # 30
    "word_freq_857",  # 31
    "word_freq_data",  # 32
    "word_freq_415",  # 33
    "word_freq_85",  # 34
    "word_freq_technology",  # 35
    "word_freq_1999",  # 36
    "word_freq_parts",  # 37
    "word_freq_pm",  # 38
    "word_freq_direct",  # 39
    "word_freq_cs",  # 40
    "word_freq_meeting",  # 41
    "word_freq_original",  # 42
    "word_freq_project",  # 43
    "word_freq_re",  # 44  reply threads
    "word_freq_edu",  # 45
    "word_freq_table",  # 46
    "word_freq_conference",  # 47
    "char_freq_;",  # 48  frequency of ';'
    "char_freq_(",  # 49  frequency of '('
    "char_freq_[",  # 50  frequency of '['
    "char_freq_!",  # 51  exclamation marks (often big)
    "char_freq_$",  # 52  dollar sign (money related)
    "char_freq_#",  # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",  # 56  total number of capital letters
    "spam_label",  # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES

X = df.drop("spam_label", axis=1)  # input features (what the model learns from)
y = df["spam_label"]  # target label (spam or not spam)

print(  # information about the dataset
    f"Dataset shape: {X.shape}\n"
    f"Number of emails: {len(X)}\n\n"
    f"First few rows:\n {X.head()}\n\n"
    f"Class distribution:\n {df['spam_label'].value_counts()}\n\n"
    f"Class balance (%):\n {y.value_counts(normalize=True).mul(100).round(2)}\n"
    # f"Feature value ranges:\n {X.describe().loc[["min", "max", "mean"]].T.to_string()}\n"
)

# The dataset is imbalanced (39.4% spam, 61.6% ham), meaning the raw accuracy score can be misleading. A model that predicts "not spam" for every email would already achieve ~61%, so it's important to look at precision and recall for the spam class.

for feature in ["word_freq_free", "char_freq_!", "capital_run_length_total"]:
    name = feature.replace("_", "-").replace("!", "exclamation")
    fig, ax = plt.subplots(figsize=(10, 6))
    spam = X.loc[y == 1, feature]  # spam values
    ham = X.loc[y == 0, feature]  # ham values
    ax.boxplot([ham, spam], tick_labels=["Ham", "Spam"])
    ax.set_title(f"{feature}\nHam vs Spam")
    ax.set_ylabel(feature)
    plt.tight_layout()
    plt.savefig(f"outputs/boxplot_{name}.png")
    plt.close()
    print(f"{feature} median --- Ham: {ham.median():.4f}, Spam: {spam.median():.4f}")

# Differences between classes are apparent. Based on these box plots showing distribution of a feature for spam vs ham emails, emails marked as spam tend to 1) contain the word "free", 2) use many exclamation marks, and 3) use long sequences of capital letters. Such patterns align with intuition and suggest that these features may be useful for classification.

# Feature numeric scales vary dramatically across the dataset: some features are small fractions, while others reach into the thousands. Models like KNN and Logistic Regression run into this problem because they rely on distances or gradients, which may allow larger-scale features to dominate the model if left unnormalized. Tree-based models don't have this issue because they split on thresholds rather than distances.

# Most word-frequency features are sparse, with many emails containing none of a particular word. This produces right-skewed distributions with many zeros, meaning only a small subset of emails contain informative word frequencies.

# ---------------------------------------------------------------------------- #
print("\n\n------ Task 2: Prepare Your Data")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 80/20 split to preserve class balance
)
print(f"Training set size: {X_train.shape}\n" f"Testing set size: {X_test.shape}\n")

# Standardize the features before PCA because PCA is based on variance. Without scaling, features with much larger numeric ranges would dominate the principal components. The scaler is fit only on the training data, then applied to the test data to prevent data leakage.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA is fit only on X_train_scaled so that information from the test set is never used when learning the principal components. The fitted PCA transformation is then applied to both the training and testing data.
pca = PCA()
pca.fit(X_train_scaled)

cum_var = np.cumsum(pca.explained_variance_ratio_)
n = int(np.argmax(cum_var >= 0.90)) + 1  # n_components

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

plt.figure(figsize=(6, 4))
plt.plot(cum_var)
plt.axhline(y=0.90, color="r", linestyle="--", label="90% threshold")
plt.axvline(x=n, color="g", linestyle="--", label=f"n={n}")
plt.title("Cumulative Explained Variance (Spambase)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

print(
    f"90% cumulative explained variance reached at {n} principal components.\n"
    f"Scaled training shape: {X_train_scaled.shape}\n"
    f"Scaled testing shape: {X_test_scaled.shape}\n"
    f"PCA-reduced training shape: {X_train_pca.shape}\n"
    f"PCA-reduced testing shape: {X_test_pca.shape}\n"
)

# ---------------------------------------------------------------------------- #
print("\n\n------ Task 3: A Classifier Comparison")

models = {
    "KNN (unscaled)": (KNeighborsClassifier(n_neighbors=5), X_train, X_test),
    "KNN (scaled)": (
        KNeighborsClassifier(n_neighbors=5),
        X_train_scaled,
        X_test_scaled,
    ),
    "KNN (PCA)": (KNeighborsClassifier(n_neighbors=5), X_train_pca, X_test_pca),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train,
        X_test,
    ),
    "Logistic Regression (scaled)": (
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
        X_train_scaled,
        X_test_scaled,
    ),
    "Logistic Regression (PCA)": (
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
        X_train_pca,
        X_test_pca,
    ),
}
for name, (model, Xtr, Xte) in models.items():  # doesn't overwrite the datasets
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    print(f"{name} -- Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

# KNN Comparison: KNN performs poorly on the unscaled data because distance calculations are dominated by features with large numeric ranges. Scaling greatly improves performance because all features contribute more equally. Applying PCA after scaling gives a small additional improvement, suggesting that removing redundant dimensions benefits KNN.

# Logistic Regression Comparison: Logistic Regression performs well on the scaled data, but reducing the features with PCA produces slightly lower accuracy. For this dataset, PCA removes some information that the linear classifier can still use, so the non-PCA version is preferred.

# Random Forest performs the best, likely due to how it combines numerous decision trees and averages their predictions. This reduces variance and increases generalization. For this particular dataset, I would use Random Forest for its higher precision on spam (ham labelled as spam, where it makes fewer false positives than false negatives) to minimize the amount of legitimate emails marked incorrectly as spam. False negatives letting some spam through can be easily dealt with through manual user intervention, which I think is a tolerable compromise.

print("---- Decision Tree Depth Comparison")
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(random_state=42, max_depth=depth)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print(
        f"max_depth={depth}: Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}"
    )

# The training accuracy approaches 1.0 as max_depth increased, but testing accuracy started to drop (as with the case of depth=None where Training Accuracy reaches 0.9995 but Testing Accuracy begins drops to 0.9186). These could be a sign of overfitting as the model starts memorizing the training data instead of leaning general patterns.

# I would choose max_depth=10 for production because it achieved the highest testing accuracy while keeping the gap between training and testing accuracy much smaller than the unrestricted tree. Although depth=None nearly memorizes the training set (99.95% accuracy), its lower testing accuracy indicates overfitting. Depth=10 provides the best balance between model complexity and generalization.

CHOSEN_DEPTH = 10
dt_final = DecisionTreeClassifier(random_state=42, max_depth=CHOSEN_DEPTH)
dt_final.fit(X_train, y_train)
dt_pred = dt_final.predict(X_test)
print(
    f"Chosen depth: {CHOSEN_DEPTH} -- Accuracy: {accuracy_score(y_test, dt_pred):.4f}\n"
    f"{classification_report(y_test, dt_pred, target_names=["Ham", "Spam"])}\n"
)

# best model confusion matrix
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
cm = confusion_matrix(y_test, rf.predict(X_test))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"]).plot(
    cmap="copper"
)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()
print("Best model confusion matrix saved.\n")

# top 10 features > decision tree and random forest > bar chart
feature_names = list(X.columns)
dt_imp = pd.Series(dt_final.feature_importances_, index=feature_names).nlargest(10)
rf_imp = pd.Series(rf.feature_importances_, index=feature_names).nlargest(10)
print(
    "---- Top 10 Most Important Features\n"
    f"Decision Tree top 10:\n {dt_imp.to_string()}\n\n"
    f"Random Forest top 10:\n {rf_imp.to_string()}\n"
)
plt.figure(figsize=(8, 6))
rf_imp.sort_values().plot(kind="barh")
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()
print("Bar chart of Random Forest importances saved.\n")

# Both decision tree and random forest rank "char_freq_$", "word_freq_remove", and 'char_freq_!' near the top as the most important features. This matches intuition, suggesting that punctuation like dollar signs and exclamation marks tend to be signs of spam.

# ---------------------------------------------------------------------------- #
print("\n\n------ Task 4: Cross-Validation")

cv_models = [
    (
        "KNN (unscaled)",
        KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
        X_train,
    ),
    (
        "KNN (scaled)",
        Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
        ),
        X_train,
    ),
    (
        "KNN (PCA)",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n)),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        X_train,
    ),
    (
        "Logistic Regression (scaled)",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
            ]
        ),
        X_train,
    ),
    (
        "Logistic Regression (PCA)",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n)),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
            ]
        ),
        X_train,
    ),
    (f"Decision Tree (depth={CHOSEN_DEPTH})", dt_final, X_train),
    ("Random Forest", rf, X_train),
]
cv_results = []
for name, model, Xtr in cv_models:
    scores = cross_val_score(model, Xtr, y_train, cv=5)
    cv_results.append((name, scores.mean(), scores.std()))
print("Cross-Validation Summary")
for name, mean, std in cv_results:
    print(f"{name:<35} --  Mean={mean:.4f}, Std={std:.4f}")

# No scaling needed on tree-based models (Decision Tree and Random Forest), and I wrapped models that need preprocessing in a Pipeline so scaler/PCA are re-fit on each fold's training portion only. Through 100 trees, Random Forest has the highest mean CV accuracy and among the lowest standard deviations. A Decision Tree has higher variance across folds, suggesting it is more sensitive to the specific training split and lacks comparable stability to a Random Forest. The ranking generally matches the single train/test split results for each classifier.

# ---------------------------------------------------------------------------- #
print("\n\n------ Task 5: Building a Prediction Pipeline")

lr_pipeline = Pipeline(  # Logistic Regression (best non-tree model)
    [
        ("scaler", StandardScaler()),  # name, object pattern
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
    ]
)
rf_pipeline = Pipeline(  # Random Forest (best tree model)
    [("classifier", RandomForestClassifier(n_estimators=100, random_state=42))]
)

# Logistic Regression was selected because it achieved the best performance among the non-tree-based models in Task 3. Random Forest was selected because it achieved the highest overall accuracy among the tree-based models.

lr_manual = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear").fit(
    X_train_scaled, y_train
)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
lr_acc = accuracy_score(y_test, lr_manual.predict(X_test_scaled))

for name, pipeline, Xtr, Xte, manual_acc in [
    ("Random Forest Pipeline", rf_pipeline, X_train, X_test, rf_acc),
    ("Logistic Regression Pipeline", lr_pipeline, X_train, X_test, lr_acc),
]:
    pipeline.fit(Xtr, y_train)
    y_pred_pipe = pipeline.predict(Xte)
    pipe_acc = accuracy_score(y_test, y_pred_pipe)
    print(
        f"{name}\nPipeline Accuracy:  {pipe_acc:.4f}\n"
        f"Manual Accuracy: {manual_acc:.4f}\n"
        f"Accuracy Difference: {abs(pipe_acc - manual_acc):.4f}\n"
        f"{classification_report(y_test, y_pred_pipe, target_names=["Ham", "Spam"])}\n"
    )

# The Random Forest pipeline only needed the classifier and no prior preprocessing/scaling. Conversely, the Logistic Regression pipeline reproduces the best non-tree workflow and requires preprocessing by applying scaling before classification, with PCA intentionally omitted (because it did not improve performance in earlier comparisons). In both pipelines, the difference should be 0.0000 to to confirm that both are using the same data and correctly reproducing the manual workflow.

# Pipelines have practical value in making workflows less prone to missing steps and more easier to maintain, especially when working with other developers. Any preprocessing (such as scaling) are applied when needed in a predictable order, ensuring scalers are fit on each fold's training data only and preventing data leakage.
