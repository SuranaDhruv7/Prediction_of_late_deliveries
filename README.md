# 🚚 Predict Late Deliveries

> 📊 **Data Science & AI/ML Practical Project**\
> 🎯 **Objective:** Analyze delivery data and prepare a machine-learning
> workflow to predict whether a delivery will be late.

------------------------------------------------------------------------

## 📌 Project Overview

**Predict Late Deliveries** is a practical Data Science project based on
delivery-related data.

The project covers:

-   🐍 Python programming
-   🐼 Data handling with Pandas
-   🔢 Numerical computing with NumPy
-   📊 Data visualization
-   📐 Statistical analysis
-   🧪 Hypothesis testing
-   📏 Confidence intervals
-   🧮 Basic linear algebra
-   🧹 Data preprocessing
-   ✂️ Train/validation/test splitting
-   🤖 Machine Learning preparation

The main target variable is **`late`**, which represents whether a
delivery was late.

------------------------------------------------------------------------

## 🎯 Project Objective

The objective is to work with delivery information and prepare the data
for a classification problem:

    `late` Meaning
  -------- --------------------------
       `0` ✅ Delivery was not late
       `1` ⚠️ Delivery was late

The dataset contains information related to:

-   📍 Delivery distance
-   📦 Load
-   🚦 Traffic
-   👥 Staff
-   🏷️ Group

------------------------------------------------------------------------

## 📂 Project Structure

``` text
📦 Predict-Late-Deliveries
│
├── 📓 exam.ipynb
├── 🐍 generate_data.py
├── 📄 set_b.csv
└── 📖 README.md
```

### 📓 `exam.ipynb`

Main notebook containing the complete practical workflow and analysis.

### 🐍 `generate_data.py`

Python script used to generate the synthetic delivery dataset.

### 📄 `set_b.csv`

Dataset used by the notebook.

### 📖 `README.md`

Project documentation.

------------------------------------------------------------------------

## 🛠️ Technologies & Libraries

The notebook imports the following libraries:

``` python
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import t
from scipy.stats import ttest_ind

from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
```

### 📚 Main Libraries

  Library           Purpose
  ----------------- ----------------------------------------------
  🐼 Pandas         Data loading and manipulation
  🔢 NumPy          Numerical and array operations
  📊 Matplotlib     Data visualization
  🎨 Seaborn        Statistical visualization
  🧪 SciPy          Statistical testing and confidence intervals
  🤖 Scikit-learn   Data splitting, preprocessing and ML
  🧠 TensorFlow     Deep learning framework

------------------------------------------------------------------------

# 📊 Dataset

The dataset is loaded using:

``` python
data = pd.read_csv("set_b.csv")
```

### Dataset Dimensions

The supplied CSV contains:

-   🔢 **305 rows**
-   📋 **7 columns**
-   🔁 **5 duplicate rows**
-   ❌ **15 missing values in `distance`**
-   ❌ **15 missing values in `load`**

After removing duplicate rows:

-   ✅ **300 unique rows**

------------------------------------------------------------------------

## 🧾 Dataset Columns

  Column        Data Type   Description
  ------------- ----------- -------------------------------
  `record_id`   Integer     Record identifier
  `distance`    Float       Delivery distance
  `load`        Float       Delivery load
  `traffic`     Float       Traffic value
  `staff`       Float       Staff value
  `group`       Object      Group/category (`G1` or `G2`)
  `late`        Integer     Target variable

------------------------------------------------------------------------

# 🧹 Section A --- Technical Work

The notebook starts the practical analysis with a series of
data-processing and statistical steps.

------------------------------------------------------------------------

## 1️⃣ Checking Duplicates

The notebook checks duplicate rows using:

``` python
data.duplicated().sum()
```

The supplied dataset contains:

``` text
5 duplicate rows
```

🔎 Duplicate records can affect statistical analysis and
machine-learning workflows, so they are removed.

------------------------------------------------------------------------

## 2️⃣ Dropping Duplicates

Duplicates are removed using:

``` python
data.drop_duplicates(inplace=True)
```

The notebook then checks again:

``` python
data.duplicated().sum()
```

After removal:

``` text
0 duplicate rows
```

The dataset changes from:

``` text
305 rows → 300 rows
```

------------------------------------------------------------------------

# ✂️ 3️⃣ Train / Test / Validation Split

The notebook first creates a training and test split:

``` python
fit, test = train_test_split(
    data,
    test_size=0.20,
    stratify=data["late"],
    random_state=42
)
```

Then the training portion is further divided into training and
validation sets:

``` python
fit, val = train_test_split(
    fit,
    test_size=0.20,
    stratify=fit["late"],
    random_state=42
)
```

### 📌 Resulting Structure

``` text
                 Dataset
                    │
          ┌─────────┴─────────┐
          │                   │
        Fit                  Test
          │
     ┌────┴────┐
     │         │
   Train      Validation
```

### 🎯 Why `stratify`?

``` python
stratify=data["late"]
```

Stratification helps maintain a similar distribution of the target
classes across the splits.

### 🔒 Why `random_state=42`?

It makes the split reproducible.

------------------------------------------------------------------------

# 📍 4️⃣ Distance Statistical Analysis

The notebook analyzes the `distance` variable.

Missing distance values are removed for this analysis:

``` python
distance = fit["distance"].dropna()
```

The notebook calculates:

### 🔢 Observed Count

``` python
distance.count()
```

### 📌 Mean

``` python
distance.mean()
```

### 📌 Median

``` python
distance.median()
```

### 📏 Sample Standard Deviation

``` python
distance.std(ddof=1)
```

Using:

``` python
ddof=1
```

calculates the **sample standard deviation**.

------------------------------------------------------------------------

# 📊 5️⃣ Distance Distribution

A histogram is created to visualize the distribution of delivery
distance:

``` python
plt.hist(distance, bins=20)

plt.title("Distribution of Distance")
plt.xlabel("Distance")
plt.ylabel("Frequency")
```

### 🎯 Purpose

The histogram helps understand:

-   📈 Distribution shape
-   📊 Frequency of observations
-   📍 Spread of distance values
-   🔎 Possible unusual values

------------------------------------------------------------------------

# 🧪 6️⃣ Welch's T-Test

The notebook compares the `distance` values between groups **G1** and
**G2**.

### Group G1

``` python
g1 = fit[
    (fit["group"] == "G1") &
    (fit["distance"].notna())
]["distance"]
```

### Group G2

``` python
g2 = fit[
    (fit["group"] == "G2") &
    (fit["distance"].notna())
]["distance"]
```

The test is performed using:

``` python
result = ttest_ind(
    g1,
    g2,
    equal_var=False
)
```

This is a **Welch's independent two-sample t-test**.

------------------------------------------------------------------------

## 📌 Test Output

The notebook prints:

``` python
print("G1 count:", len(g1))
print("G2 count:", len(g2))
print("T-statistic:", result.statistic)
print("P-value:", result.pvalue)
```

### 🧠 Important Parameters

  Parameter           Meaning
  ------------------- ------------------------------------
  `g1`                Distance values for G1
  `g2`                Distance values for G2
  `equal_var=False`   Uses Welch's version of the t-test
  `statistic`         Calculated t-statistic
  `pvalue`            p-value of the test

> 📌 The notebook contains the test implementation, but this README does
> not claim a statistical conclusion because the supplied notebook does
> not contain a saved output for the test result.

------------------------------------------------------------------------

# 📐 7️⃣ 95% Confidence Interval

The notebook calculates a 95% confidence interval for the mean distance.

First:

``` python
n = len(distance)
mean = distance.mean()
sem = distance.sem()
```

Then:

``` python
ci = t.interval(
    0.95,
    df=n-1,
    loc=mean,
    scale=sem
)
```

Finally:

``` python
print("95% CI:", ci)
```

### 🧠 Parameters

  Parameter     Meaning
  ------------- ----------------------------
  `0.95`        95% confidence level
  `df=n-1`      Degrees of freedom
  `loc=mean`    Sample mean
  `scale=sem`   Standard error of the mean

------------------------------------------------------------------------

# 🧮 8️⃣ Linear Algebra

The notebook selects two numerical variables:

``` python
X = fit[["distance", "traffic"]].dropna().values
```

The resulting data is converted into a NumPy array using:

``` python
.values
```

The number of complete observations is printed:

``` python
print("n =", len(X))
```

### 📌 Selected Features

``` text
distance
traffic
```

This section demonstrates basic preparation of tabular data for
numerical and linear-algebra operations.

------------------------------------------------------------------------

# 🔍 9️⃣ Data Preprocessing Checks

The notebook checks the cleaned dataset using:

``` python
print("Shape:", data.shape)
print("Duplicates:", data.duplicated().sum())

print("Missing values:")
print(data.isna().sum())
```

This provides information about:

-   📐 Dataset shape
-   🔁 Duplicate records
-   ❌ Missing values

------------------------------------------------------------------------

## 🧹 Creating a Clean Dataset

The notebook creates:

``` python
clean = data.drop_duplicates()
```

Then checks:

``` python
print("Clean shape:", clean.shape)
print("Unique IDs:", clean["record_id"].nunique())
```

This verifies the cleaned dataset and uniqueness of record IDs.

------------------------------------------------------------------------

# 🎯 🔟 Assigning the Target Variable

The target variable is separated from the input data:

``` python
x = data.drop("late", axis=1)
y = data["late"]
```

### `x`

Contains all features except `late`.

### `y`

Contains the target variable:

``` text
late
```

------------------------------------------------------------------------

# ✂️ 1️⃣1️⃣ Train-Test Split

The notebook also performs a direct feature/target train-test split:

``` python
x_test, x_train, y_test, y_train = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 🎯 Parameters

  Parameter        Value   Purpose
  ---------------- ------- -----------------------------------
  `test_size`      `0.2`   20% test portion
  `random_state`   `42`    Reproducibility
  `stratify`       `y`     Preserve target-class proportions

> ⚠️ The notebook contains both the earlier `fit/test/val` split and
> this later `x/y` split. This README documents both exactly as
> implemented.

------------------------------------------------------------------------

# 🤖 Machine Learning Components

The notebook imports several Machine Learning components for the
practical workflow.

## 📈 Logistic Regression

``` python
from sklearn.linear_model import LogisticRegression
```

Used for classification problems where the target has two classes.

------------------------------------------------------------------------

## 🧪 Dummy Classifier

``` python
from sklearn.dummy import DummyClassifier
```

Provides a simple baseline for comparing classification performance.

------------------------------------------------------------------------

## 🧩 K-Means

``` python
from sklearn.cluster import KMeans
```

Used for unsupervised clustering.

------------------------------------------------------------------------

## 📏 StandardScaler

``` python
from sklearn.preprocessing import StandardScaler
```

Used to standardize numerical features.

------------------------------------------------------------------------

## 🔤 OneHotEncoder

``` python
from sklearn.preprocessing import OneHotEncoder
```

Used to convert categorical variables into numerical representations.

For this dataset, the categorical variable is:

``` text
group
```

with values such as:

``` text
G1
G2
```

------------------------------------------------------------------------

# 📊 Classification Metrics

The notebook imports four common classification metrics.

### 🎯 Accuracy

``` python
accuracy_score
```

Measures the overall proportion of correct predictions.

### 🎯 Precision

``` python
precision_score
```

Measures the proportion of predicted positive cases that are actually
positive.

### 🎯 Recall

``` python
recall_score
```

Measures how many actual positive cases are correctly identified.

### 🎯 F1 Score

``` python
f1_score
```

Combines precision and recall into a single metric.

------------------------------------------------------------------------

# 🧪 Dataset Generation

The `generate_data.py` script creates the dataset programmatically.

The script:

1.  🔢 Creates **300 observations**.
2.  📊 Defines four numerical columns:
    -   `distance`
    -   `load`
    -   `traffic`
    -   `staff`
3.  🏷️ Generates two groups:
    -   `G1`
    -   `G2`
4.  🧮 Calculates an underlying score.
5.  🎯 Converts the score into the binary `late` target.
6.  🆔 Creates `record_id`.
7.  ❌ Adds missing values to `distance` and `load`.
8.  🔁 Adds five duplicate rows.
9.  💾 Saves the data to a CSV file.

The dataset-generation script uses a fixed NumPy random seed:

``` python
rng = np.random.default_rng(202)
```

This makes the generated data reproducible.

------------------------------------------------------------------------

# 🔄 Complete Project Workflow

``` text
🚀 Start
   │
   ▼
📥 Load Dataset
   │
   ▼
🔍 Inspect Dataset
   │
   ▼
🔁 Check Duplicates
   │
   ▼
🧹 Remove Duplicates
   │
   ▼
✂️ Split Data
   │
   ├── Training
   ├── Validation
   └── Testing
   │
   ▼
📍 Analyze Distance
   │
   ├── Mean
   ├── Median
   └── Standard Deviation
   │
   ▼
📊 Plot Distance Distribution
   │
   ▼
🧪 Welch's T-Test
   │
   ▼
📐 95% Confidence Interval
   │
   ▼
🧮 Linear Algebra Preparation
   │
   ▼
🧹 Preprocessing Checks
   │
   ▼
🎯 Separate Features & Target
   │
   ▼
🤖 Machine Learning Preparation
   │
   ▼
📊 Evaluation Metrics
```

------------------------------------------------------------------------

# 💡 Key Concepts Demonstrated

This practical demonstrates the following Data Science concepts:

### 🐍 Python

-   NumPy arrays
-   Pandas DataFrames
-   Data manipulation

### 🧹 Data Cleaning

-   Duplicate detection
-   Duplicate removal
-   Missing-value inspection
-   Unique ID checking

### 📊 Statistics

-   Mean
-   Median
-   Sample standard deviation
-   Standard error
-   Confidence interval

### 🧪 Hypothesis Testing

-   Independent two-sample t-test
-   Welch's t-test
-   p-value
-   t-statistic

### 📈 Visualization

-   Histogram
-   Distribution analysis

### 🧮 Linear Algebra

-   Feature matrix creation
-   NumPy array conversion

### 🤖 Machine Learning

-   Train/test splitting
-   Stratified splitting
-   Feature/target separation
-   Standardization
-   One-hot encoding
-   Classification models
-   Clustering

### 📊 Model Evaluation

-   Accuracy
-   Precision
-   Recall
-   F1 Score

------------------------------------------------------------------------

# 📌 Dataset Summary

  Property                                 Value
  --------------------------------- ------------
  📊 Raw rows                                305
  📋 Columns                                   7
  🔁 Duplicate rows                            5
  ✅ Rows after duplicate removal            300
  ❌ Missing `distance`                       15
  ❌ Missing `load`                           15
  🎯 Target                               `late`
  🏷️ Categorical feature                 `group`
  🧪 Groups                           `G1`, `G2`

------------------------------------------------------------------------

# 🚀 Future Scope

The imported Machine Learning components provide a foundation for
extending the project into a complete predictive modeling pipeline.

Possible next steps include:

-   🧹 Handle missing values explicitly.
-   🔤 Encode the `group` feature.
-   📏 Scale numerical features.
-   🤖 Train Logistic Regression.
-   🧪 Establish a Dummy Classifier baseline.
-   📊 Evaluate models using Accuracy, Precision, Recall and F1 Score.
-   🔍 Compare different models.
-   🎯 Tune model parameters.
-   📈 Add additional visualizations.
-   🚚 Build a final late-delivery prediction pipeline.

> ⚠️ These are future extensions. The supplied notebook imports these ML
> components but does not contain a completed model-training/evaluation
> section after the final split.

------------------------------------------------------------------------

# ▶️ How to Run

## 1️⃣ Install Python

Make sure Python is installed on your system.

## 2️⃣ Install Required Libraries

``` bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow
```

## 3️⃣ Generate the Dataset

Run:

``` bash
python generate_data.py
```

The script generates the delivery dataset.

## 4️⃣ Open the Notebook

Open:

``` text
exam.ipynb
```

using:

-   📓 Jupyter Notebook
-   💻 VS Code
-   🧪 JupyterLab

## 5️⃣ Run the Cells

Execute the notebook cells from top to bottom to reproduce the analysis.

------------------------------------------------------------------------

# 🎥 Project Explanation

You can watch the project explanation video here:

🔗 **[Project Explanation Video](https://drive.google.com/file/d/1KIe9IKjvgOtABnEUqmioc5Yhn8-_TceN/view?usp=drive_link)**

# 👨‍💻 Author

## Dhruv Surana

🎯 Aspiring Data Scientist\
💻 Python \| SQL \| Excel \| Power BI \| Machine Learning \| Data
Science

------------------------------------------------------------------------

# ⭐ Project Summary

**Predict Late Deliveries** demonstrates a practical Data Science
workflow starting from raw delivery data and moving through data
cleaning, statistical analysis, visualization, preprocessing, and
Machine Learning preparation.

``` text
📥 Data
  ↓
🧹 Cleaning
  ↓
📊 Analysis
  ↓
🧪 Statistics
  ↓
🧮 Feature Preparation
  ↓
🤖 ML Preparation
  ↓
🎯 Late Delivery Prediction
```

------------------------------------------------------------------------

## 📜 License

This project is created for **educational, practical examination, and
portfolio purposes**. 🚀
