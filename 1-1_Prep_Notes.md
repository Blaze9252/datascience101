# 1-1 Assessment Prep Notes

Instructors in a 1-1 rarely ask "what does `.head()` do?". They ask **"why did you do that?"** and **"what does this number mean?"**. Answer in your own words, and point at the notebook output when you can.

---

## Part 1: Linear regression (car prices)

**"Walk me through what you did."**
I loaded the car dataset and looked at its structure with `head`, `info` and `describe`. I made sure horsepower had no missing values, filling any with the mean. I explored relationships with a correlation heatmap and a pairplot. Then I removed columns that don't help prediction and turned the text columns into numbers with dummy variables. I split the data 80/20 into train and test, scaled the features, trained a linear regression and checked it with R² and RMSE.

**"Why drop car_ID, symboling and CarName?"**
- **car_ID** is just a row number, so it has no link to price.
- **CarName** has 147 unique values across 205 cars. As dummies it would add a huge number of columns and the model would memorise the data instead of learning from it (overfitting).
- **symboling** is an insurance risk rating. The template said to drop it, but you can fairly say it could be kept as a feature. Admitting that shows you're thinking, not copying.

**"What are dummy variables and why do you need them?"**
Models only understand numbers. `get_dummies` turns a text column like fueltype (gas or diesel) into 0/1 columns, one per category. If you instead labelled sedan=1, hatchback=2, convertible=3, the model would think a convertible is "three times" a sedan. Dummies avoid giving categories a fake order.

**"Why did you scale the features?"**
The features are on very different scales: curbweight is in the thousands, boreratio is around 3. StandardScaler gives every feature a mean of 0 and a standard deviation of 1, so they're comparable. For plain linear regression, scaling doesn't change the predictions, but it makes the coefficients comparable. It matters a lot for KNN and logistic regression.

**"Why `fit_transform` on train but only `transform` on test?"** This one comes up a lot.
The scaler should learn the mean and standard deviation from the training data only. If it also learned from the test data, information from the test set would leak into training. That's called **data leakage**, and it makes your results look better than they really are. The test set has to stand in for data the model has never seen.

**"What do your results mean?"**

| Metric | Value | Meaning |
|---|---|---|
| R² | 0.86 | The model explains about 86% of the variation in car price. Good. |
| RMSE | ≈ 3,284 | Predictions are typically off by about 3,284. The average car costs about 13,277, so that's roughly 25% error. Decent, not precise. |
| MSE | ≈ 10.8 million | RMSE squared. Hard to interpret on its own, which is why you take the square root. |

**"Anything wrong with your predictions?"** Raise this yourself. It impresses.
One prediction is **negative (about -1,954)**. A car can't have a negative price. Linear regression has no built-in limits, so it can predict impossible values. Ways to fix it: predict log(price) instead of price, or use a model like a random forest.

**"What does the correlation heatmap tell you?"**
- The strongest positive correlations with price: enginesize (0.87), curbweight (0.84), horsepower (0.81), carwidth (0.76).
- citympg (-0.69) and highwaympg (-0.70) are negative: more fuel-efficient cars tend to be cheaper.
- **Multicollinearity:** citympg and highwaympg are 0.97 correlated with each other, so they carry nearly the same information.

---

## Part 2: Classification (breast cancer)

**"What's the dataset and target?"**
It's the scikit-learn breast cancer dataset: 569 tumours, 30 measurements each, and a yes/no label. **0 = malignant (cancer), 1 = benign.** Know this, because everything below depends on it.

**"Why did the three models perform differently?"**

| Model | Accuracy | Why |
|---|---|---|
| Logistic regression | 96.5% | A linear model, and this data separates well with a straight-line boundary once it's scaled. |
| KNN | 95.6% | Classifies a point by a vote of its nearest neighbours. Depends on distances, so it needs scaled data. |
| Decision tree | 93.9% | Splits the data with yes/no rules. A single tree tends to overfit, which is why it scored lowest. |

**"How do you read the confusion matrix?"**
Rows are the **actual** class and columns are the **predicted** class. Take KNN's matrix:

```
              Predicted 0   Predicted 1
Actual 0 (M)      42            5        ← 5 cancers missed (false negatives)
Actual 1 (B)       0           67
```

- 42 malignant tumours were correctly caught.
- **5 malignant tumours were predicted benign.** These are false negatives, the dangerous error.
- 0 benign tumours were wrongly flagged as malignant.
- 67 benign tumours were correctly identified.

**"So which model is best?"** This is the strongest point you can make.
Accuracy isn't the whole story. In cancer detection, missing a cancer is far worse than a false alarm. So **recall on the malignant class** matters most. KNN misses 5 cancers, while logistic regression and the decision tree each miss 2. Logistic regression wins on both accuracy and safety. KNN has 95.6% accuracy but the worst result where it matters.

**"What is GridSearchCV doing?"**
It tries every combination of the hyperparameters you list. For each combination it runs **5-fold cross-validation**: it splits the training data into 5 parts, trains on 4 and validates on 1, rotating through all 5. Then it keeps the combination with the best average score. This avoids tuning to one lucky split. The test set is never touched during tuning.

**"What are the hyperparameters you tuned?"**
- **Decision tree:** `max_depth` (how deep the tree can grow, which controls overfitting) and `criterion` (gini or entropy, two ways of measuring how mixed a split is).
- **Logistic regression:** `C`, the inverse of regularisation strength. A small C means a simpler, more constrained model. Also `penalty` (L1, L2 or none) and `solver` (the optimisation algorithm).
- **KNN:** `n_neighbors`, how many neighbours get a vote.

**"Precision vs recall?"**
- **Precision:** of everything I predicted as positive, how much was actually positive?
- **Recall:** of everything that was actually positive, how much did I catch?
- **F1-score:** a balance of the two.

---

## Part 3: Changes from the template (own them if asked)

- I used `None` instead of `'none'` for the logistic regression penalty, because newer scikit-learn removed the string option.
- I added `numeric_only=True` to `corr()`, because pandas 3 crashes on text columns otherwise.
- I filled horsepower with an assignment instead of `inplace=True`, because `inplace` on a single column no longer works in pandas 3.
- I scaled the classification data, because logistic regression and KNN depend on scale.

---

## Part 4: Quick-fire theory from the written paper

- **Correlation:** ranges from -1 to +1, measures the strength and direction of a linear relationship. Correlation is not causation.
- **Linear regression:** fits the straight line that minimises the squared errors (Ordinary Least Squares).
- **Classification vs regression:** classification predicts categories, regression predicts continuous numbers.
- **EDA (exploratory data analysis):** understand the data, find problems and spot patterns before modelling.
- **Overfitting:** the model memorises the training data and does badly on new data. Signs: high train accuracy, low test accuracy. Fixes: a simpler model, regularisation, more data, cross-validation.
- **NumPy array vs list:** an array holds a single data type and is fast and vectorised, so it can do maths on the whole array at once.
- **Reverse a string:** `s[::-1]`. **Last element of a list:** `lst[-1]`.

---

**Before the session:** open `Practical_Assessment.ipynb` and run it cell by cell yourself. Being able to point at an output and explain it counts for more than reciting definitions.
