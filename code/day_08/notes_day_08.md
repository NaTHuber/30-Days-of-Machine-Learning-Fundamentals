# Day 08. Data splitting: train/test
## Today's objective 
Learn why and how to split a dataset into training and testing subsets to evaluate machine learning models fairly.

## Motivation 
When you build a model, you want to know how well it performs on **unseen** data. If you train and test on the same dataset, the model might just “memorize” the data (overfitting).

That’s why we split the dataset:

- **Training set:** used to fit (train) the model.
- **Testing set:** used to evaluate the model’s performance on new data.

The typical split is 70–80% for training and 20–30% for testing, but this can vary.

In Python, you can use `train_test_split` from `sklearn.model_selection`.

## Example 
```python 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load a sample dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

```
**Notes:**
- `test_size=0.2` → 20% test set.
- `random_state` ensures reproducibility.
- `stratify=y` keeps class proportions the same in train and test sets (important for classification).

Splitting data is a fundamental step in machine learning. The simplest and most common approach is to divide the dataset into training and testing sets. The training set is where the model learns, while the testing set is kept aside to evaluate how well the model generalizes to unseen data. This practice prevents us from being misled by overly optimistic results when a model performs well only on the data it has already seen (overfitting). In this way, the test set becomes our “final exam” to measure true performance.

## Small note 

> In more advanced projects, you may also encounter a third set: the **validation set**. This extra split is particularly useful when tuning hyperparameters or making design choices, because it allows you to optimize your model without accidentally adapting it to the test set. By keeping the test set untouched until the very end, you preserve its role as an more _honest_ measure of performance 🌱
