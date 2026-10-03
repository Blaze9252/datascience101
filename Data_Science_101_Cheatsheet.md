# Data Science 101 Cheatsheet

A simplified guide to learn, understand and remember the core ideas. Read it top to bottom once, then use the sections as a reference.

---

## 1. The big picture in one paragraph

Data science is using data to answer a question or make a prediction. You **collect** data, **clean** it, **explore** it to understand it, turn it into numbers a model can use, **train** a model on part of it, and **test** the model on the part it hasn't seen. If the model does well on unseen data, you can trust it on new data in the real world.

---

## 2. The data science life cycle

**Memory hook: "Pretty Cats Can Eat Fish, Making Everyone Delighted"**

| # | Step | What you do | Question you're answering |
|---|---|---|---|
| 1 | **P**roblem | Define the goal and how you'll measure success | What are we trying to predict or learn? |
| 2 | **C**ollect | Get data from files, databases, APIs, scraping | Where is the data? |
| 3 | **C**lean | Fix missing values, duplicates, wrong types, outliers | Is the data trustworthy? |
| 4 | **E**xplore (EDA) | Stats and charts to find patterns | What does the data tell us? |
| 5 | **F**eatures | Encode text, scale numbers, pick useful columns | What should the model learn from? |
| 6 | **M**odel | Choose an algorithm and train it | Which model fits the problem? |
| 7 | **E**valuate | Score it on test data, tune it | Is it good enough? |
| 8 | **D**eploy | Put it to use, monitor it, retrain when needed | How do we use it in the real world? |

---

## 3. Which type of problem is it?

```
Do you have a target column you want to predict?
│
├── YES → Supervised learning
│         │
│         ├── Target is a NUMBER (price, temperature, sales)  → REGRESSION
│         └── Target is a CATEGORY (yes/no, spam, cancer)     → CLASSIFICATION
│
└── NO  → Unsupervised learning
          └── Find groups or structure (customer segments)    → CLUSTERING
```

**Memory hook:** Regression answers **"how much?"**, classification answers **"which one?"**, clustering answers **"what groups exist?"**

---

## 4. Python basics you'll use constantly

```python
import numpy as np            # import a module with a short nickname (alias)
from math import sqrt         # import one specific function

s = "hello"
s[::-1]                       # 'olleh'  → reverse a string (step backwards)
lst = [10, 20, 30]
lst[0]                        # 10 → first element (Python counts from 0)
lst[-1]                       # 30 → last element (negative counts from the end)
lst[0:2]                      # [10, 20] → slice: start included, end excluded

a = np.array([1, 2, 3])       # NumPy array: one data type, fast maths
a * 2                         # array([2, 4, 6]) → maths on every element at once
```

**List vs NumPy array:** a list can mix types and is slow for maths. An array is a single type and does maths on the whole thing in one go (this is called "vectorised").

---

## 5. Pandas essentials (the toolkit for tables)

A **DataFrame** is a table (rows and columns). A **Series** is one column.

| Task | Code | Notes |
|---|---|---|
| Load a CSV | `df = pd.read_csv("file.csv")` | |
| First 5 rows | `df.head()` | `df.tail()` for the last 5 |
| Column types and missing values | `df.info()` | First thing to run on new data |
| Summary statistics | `df.describe()` | Mean, std, min, max, quartiles |
| Rows and columns count | `df.shape` | Returns (rows, columns) |
| Unique values | `df["col"].unique()` | `.nunique()` counts them |
| Value counts | `df["col"].value_counts()` | How often each value appears |
| Count missing per column | `df.isnull().sum()` | |
| Select a column | `df["price"]` | |
| Filter rows | `df[df["price"] > 20000]` | |
| Drop columns | `df.drop(["a", "b"], axis=1)` | axis=1 means columns, axis=0 means rows |
| Fill missing values | `df["col"] = df["col"].fillna(df["col"].mean())` | |
| Change type | `df["col"] = df["col"].astype(float)` | |
| Correlations | `df.corr(numeric_only=True)` | |
| Group and summarise | `df.groupby("carbody")["price"].mean()` | Average price per body type |

**Memory hook for `axis`:** axis=0 goes **down** the rows, axis=1 goes **across** the columns. "1 is across."

---

## 6. Typical processes, step by step

### Process A: Load and inspect new data

1. `df = pd.read_csv(...)` loads it.
2. `df.head()` shows what the data looks like.
3. `df.shape` tells you how big it is.
4. `df.info()` shows the types and any missing values.
5. `df.describe()` gives the ranges and shows anything odd (a negative age, for example).
6. `df["col"].unique()` checks the categories for typos (the car data has "toyouta" and "vokswagen").

### Process B: Clean the data

| Problem | How to spot it | Typical fix |
|---|---|---|
| Missing values | `df.isnull().sum()`, or odd placeholders like `"?"` | Fill numbers with the mean or median, categories with the most common value. Drop the row if very few are missing. |
| Wrong type | `df.info()` shows a number column as `object` (text) | `.replace("?", np.nan)` then `.astype(float)` |
| Duplicates | `df.duplicated().sum()` | `df = df.drop_duplicates()` |
| Outliers | Box plot, or `describe()` max far from the mean | Check whether it's real. Remove it, cap it, or keep it. |
| Useless columns | IDs, row numbers, free text with too many values | `df.drop([...], axis=1)` |
| Typos in categories | `.unique()` | `.replace({"toyouta": "toyota"})` |

**Mean vs median for filling:** use the median if the column has outliers, because the median isn't pulled around by extreme values.

### Process C: Exploratory data analysis (EDA)

**Goal:** understand the data before modelling. Find patterns, relationships and problems.

| Want to see... | Chart | Code |
|---|---|---|
| Distribution of one column | Histogram | `plt.hist(df["price"])` |
| Outliers | Box plot | `sns.boxplot(x=df["price"])` |
| Relationship between two numbers | Scatter plot | `plt.scatter(df["horsepower"], df["price"])` |
| All correlations at once | Heatmap | `sns.heatmap(df.corr(numeric_only=True), annot=True)` |
| Every pair of columns | Pairplot | `sns.pairplot(df)` |
| Compare categories | Bar chart | `df.groupby("carbody")["price"].mean().plot(kind="bar")` |

**Reading correlation (r):**

```
-1 ─────────── 0 ─────────── +1
perfect        no           perfect
negative     relationship   positive
(one up, other down)        (both go up together)
```

Roughly: above 0.7 (or below -0.7) is strong, 0.3 to 0.7 is moderate, below 0.3 is weak. **Correlation is not causation.** Ice cream sales and drownings both rise in summer, but one doesn't cause the other.

### Process D: Feature engineering (getting data model-ready)

**1. Encode text into numbers.** Models only understand numbers.

```python
dummies = pd.get_dummies(df[["fueltype", "carbody"]])   # one 0/1 column per category
df = pd.concat([df, dummies], axis=1)                    # stick them onto the table
df = df.drop(["fueltype", "carbody"], axis=1)            # remove the original text columns
```

**2. Split into X and y.** X is the inputs, y is what you want to predict.

```python
X = df.drop("price", axis=1)   # everything except the target
y = df["price"]                # the target
```

**3. Train/test split.** Hide some data so you can test the model honestly.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# test_size=0.2 → 80% to train, 20% to test
# random_state=0 → the same split every run, so results are reproducible
```

**4. Scale the features.** This puts every feature on a similar scale.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # LEARN the mean and std from train, then apply them
X_test = scaler.transform(X_test)         # only APPLY to test, never learn from it
```

**Golden rule:** `fit` only on training data. Fitting on test data is **data leakage**, and it makes your model look better than it really is.

**Does the model need scaling?**

| Needs scaling | Doesn't need it |
|---|---|
| KNN, logistic regression, SVM, neural networks | Decision tree, random forest |

Plain linear regression gives the same predictions either way, but scaling makes its coefficients comparable.

### Process E: Regression workflow (predict a number)

**The scikit-learn pattern. Memory hook: "I Fit, I Predict, I Score."**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

model = LinearRegression()             # 1. create the model
model.fit(X_train, y_train)            # 2. train it on training data
pred = model.predict(X_test)           # 3. predict on unseen data
print(r2_score(y_test, pred))          # 4. score it: compare predictions with true answers
print(np.sqrt(mean_squared_error(y_test, pred)))   # RMSE
```

**Regression metrics:**

| Metric | What it means | Good is... |
|---|---|---|
| **R²** | % of the variation in the target that the model explains | Closer to 1. 0.86 means 86% explained. |
| **MSE** | Average of the squared errors | Lower. Hard to read, because it's in squared units. |
| **RMSE** | Square root of MSE: the typical error, in the target's own units | Lower. Compare it with the average target value. |
| **MAE** | Average absolute error | Lower. Less sensitive to big misses than RMSE. |

**How linear regression works:** it finds the straight line `y = b0 + b1·x1 + b2·x2 + ...` that makes the total squared error as small as possible. This is called Ordinary Least Squares. b0 is the starting point (the intercept). Each b tells you how much y changes when that feature goes up by 1.

### Process F: Classification workflow (predict a category)

Same pattern, different models and metrics:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(accuracy_score(y_test, pred))          # % predicted correctly
print(confusion_matrix(y_test, pred))        # where it got things right and wrong
print(classification_report(y_test, pred))   # precision, recall and F1 for each class
```

**Common classifiers in plain English:**

| Model | How it thinks | Strengths | Weaknesses |
|---|---|---|---|
| **Logistic regression** | Draws a line (boundary) between the classes and gives a probability | Simple, fast, easy to explain | Struggles if the boundary isn't roughly straight |
| **Decision tree** | Asks a series of yes/no questions | Easy to visualise, no scaling needed | Overfits easily |
| **Random forest** | Many trees vote | Accurate, resists overfitting | Harder to explain |
| **KNN** | "You are like your nearest neighbours" | Simple, no training step | Slow on big data, needs scaling |

### Process G: Hyperparameter tuning

**Parameters** are what the model learns from the data (for example, regression coefficients). **Hyperparameters** are the settings *you* choose before training (for example, the tree's depth).

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {"max_depth": range(1, 11), "criterion": ["gini", "entropy"]}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5)
grid.fit(X_train, y_train)     # tries all 20 combinations, each with 5-fold cross-validation
print(grid.best_params_)       # the winning settings
pred = grid.predict(X_test)    # predicts using the best model
```

**Cross-validation (cv=5)** splits the training data into 5 chunks. It trains on 4 and tests on 1, rotating 5 times, and averages the scores. This is fairer than trusting a single split.

| Model | Key hyperparameters |
|---|---|
| Decision tree | `max_depth`, `criterion` |
| Logistic regression | `C` (small = simpler model), `penalty` (l1/l2), `solver` |
| KNN | `n_neighbors` |
| Random forest | `n_estimators` (number of trees), `max_depth` |

---

## 7. The confusion matrix, simplified

```
                  PREDICTED
                  No        Yes
ACTUAL   No   │   TN    │   FP    │  ← false alarm (Type I error)
         Yes  │   FN    │   TP    │
                  ↑
         missed it (Type II error)
```

- **TP** (true positive): said yes, and it was yes.
- **TN** (true negative): said no, and it was no.
- **FP** (false positive): said yes, but it was no. A false alarm.
- **FN** (false negative): said no, but it was yes. A miss.

**Memory hook:** the second word is what the model **said**. The first word is whether it was **right**. "False Negative" means it said negative, and that was wrong.

**Metrics from the matrix:**

| Metric | Formula | Plain English | Use when... |
|---|---|---|---|
| **Accuracy** | (TP+TN) / all | How often it's right overall | The classes are balanced |
| **Precision** | TP / (TP+FP) | When it says yes, how often is it right? | False alarms are costly (spam filters) |
| **Recall** | TP / (TP+FN) | Of all the real yeses, how many did it catch? | Misses are costly (cancer, fraud) |
| **F1** | Balance of the two | One number that combines precision and recall | You need both |

**Why accuracy can lie:** if 99% of transactions aren't fraud, a model that always says "not fraud" scores 99% accuracy and catches zero fraud. Check recall.

---

## 8. Overfitting vs underfitting

| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| What happens | Too simple, misses the pattern | Learns the real pattern | Memorises the noise |
| Train score | Low | High | Very high |
| Test score | Low | High | **Much lower than train** |
| Fix | More complex model, better features | Keep it | Simpler model, regularisation, more data, cross-validation |

**Memory hook:** overfitting is a student who memorised last year's exam answers. Perfect on the practice paper, fails the real one.

---

## 9. Common mistakes to avoid

1. Fitting the scaler (or anything else) on the test data. That's data leakage.
2. Judging a model on the data it trained on. Always score on the test set.
3. Trusting accuracy alone when the classes are imbalanced.
4. Leaving ID columns in as features.
5. Label-encoding unordered categories (sedan=1, hatchback=2), which invents a fake order. Use dummies.
6. Confusing correlation with causation.
7. Forgetting `random_state`, so results change on every run.
8. Not looking at the data first. Always run `head`, `info` and `describe`.

---

## 10. Plain-English glossary

| Term | Meaning |
|---|---|
| **Feature** | An input column the model learns from (X) |
| **Target / label** | The column you want to predict (y) |
| **Training set** | The data the model learns from |
| **Test set** | Hidden data used to check the model honestly |
| **Model** | A maths rule learned from data that makes predictions |
| **Supervised learning** | Learning from examples that have the right answers |
| **Unsupervised learning** | Finding patterns without answers (clustering) |
| **EDA** | Exploratory data analysis: getting to know the data |
| **Dummy variable** | A 0/1 column that represents one category |
| **Scaling** | Putting features on a similar range |
| **Data leakage** | Test information sneaking into training |
| **Hyperparameter** | A model setting you choose before training |
| **Cross-validation** | Testing on several rotating splits for a fairer score |
| **Regularisation** | A penalty that keeps a model simple to prevent overfitting |
| **Multicollinearity** | Two features so correlated that they carry the same information |
| **Jupyter Notebook** | A browser document that mixes code, output, charts and notes |
| **Matplotlib / Seaborn** | Python plotting libraries. Seaborn is a prettier layer on top of Matplotlib. |

---

## 11. The one-page summary

1. **Understand the question.** Is it regression, classification or clustering?
2. **Look at the data:** `head`, `info`, `describe`.
3. **Clean it:** missing values, types, duplicates, useless columns.
4. **Explore it:** heatmap, scatter plots, distributions.
5. **Prepare it:** dummies for text, split X/y, train/test split, scale (fit on train only).
6. **Model it:** create → fit → predict → score.
7. **Tune it:** GridSearchCV with cross-validation.
8. **Judge it:** R²/RMSE for regression. Accuracy, confusion matrix and recall for classification.
9. **Explain it:** what do the numbers mean in the real world?
