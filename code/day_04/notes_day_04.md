# Day 04. Libraries: NumPy, Pandas, Scikit-learn (Deep Dive)
## Today's objetive 
Understand and practice NumPy, Pandas, and Scikit-learn, the core Python libraries for data manipulation and analysis. 

---

## NumPy – Numerical Python
**Purpose:** Fast numerical computation using arrays and matrices.

**Why it’s important in ML:** Almost every ML library (including Pandas and Scikit-learn) uses NumPy arrays under the hood.

**Key features:**
- _N-dimensional arrays (ndarray):_ NumPy’s core data structure is the ndarray — a grid of values, all of the same type, indexed by a tuple of integers. Unlike Python lists, NumPy arrays store data in contiguous memory and use vectorized operations, making them much faster.
- _Broadcasting for arithmetic:_ A powerful mechanism that allows NumPy to perform operations on arrays of different shapes without explicitly copying or looping. Smaller arrays are “broadcast” over larger ones so their shapes match.
- _Linear algebra:_ NumPy includes optimized routines for linear algebra
- _Random number generation:_ NumPy’s random module can generate arrays of random numbers from various distributions. Essential for creating synthetic datasets, initializing model weights, splitting train/test sets, and applying random shuffling.
- _Boolean masking and indexing:_ Essential for creating synthetic datasets, initializing model weights, splitting train/test sets, and applying random shuffling.

| Category         | Common Functions / Attributes                                 | Example                         |
| ---------------- | ------------------------------------------------------------- | ------------------------------- |
| Create arrays    | `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace` | `np.zeros((2,3))`               |
| Inspect          | `.shape`, `.ndim`, `.size`, `.dtype`                          | `arr.shape`                     |
| Math ops         | `+`, `-`, `*`, `/`, `np.sqrt`, `np.exp`, `np.mean`, `np.sum`  | `np.mean(arr)`                  |
| Indexing/Slicing | `arr[0, :]`, `arr[:, 1]`, boolean masks                       | `arr[arr > 5]`                  |
| Reshape          | `.reshape()`, `.flatten()`                                    | `arr.reshape(3,2)`              |
| Linear algebra   | `np.dot`, `np.linalg.inv`, `np.linalg.eig`                    | `np.dot(a, b)`                  |
| Random           | `np.random.rand`, `np.random.randn`, `np.random.randint`      | `np.random.randint(1,10,(3,3))` |

### Example 

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape:", arr.shape)
print("Mean:", np.mean(arr))
print("Random integers:\n", np.random.randint(0, 10, (2, 3)))

```
---

##  Pandas – Data Manipulation & Analysis
**Purpose:** Handle tabular or time-series data efficiently.

**Why it’s important in ML:** Most real-world ML work involves messy tabular datasets, and Pandas is the go-to for cleaning and organizing them.

**Key features:**

- Series (1D) and DataFrame (2D)
- Reading/writing data (read_csv, to_csv, read_excel, etc.)
- Selection & filtering (.loc, .iloc)
- Aggregation and grouping (groupby)
- Handling missing data (.dropna, .fillna)

| Category        | Common Functions / Attributes                     | Example                               |
| --------------- | ------------------------------------------------- | ------------------------------------- |
| Create          | `pd.Series`, `pd.DataFrame`                       | `pd.DataFrame({'A':[1,2],'B':[3,4]})` |
| Read/Write      | `pd.read_csv`, `to_csv`, `read_excel`             | `pd.read_csv("file.csv")`             |
| Inspect         | `.head()`, `.tail()`, `.info()`, `.describe()`    | `df.head()`                           |
| Select data     | `.loc[]` (labels), `.iloc[]` (index), `df['col']` | `df.loc[0, 'A']`                      |
| Filter          | Boolean indexing                                  | `df[df['A'] > 2]`                     |
| Missing data    | `.isna()`, `.dropna()`, `.fillna()`               | `df.dropna()`                         |
| Group/Aggregate | `.groupby()`, `.agg()`                            | `df.groupby('col').mean()`            |
| Sort            | `.sort_values()`, `.sort_index()`                 | `df.sort_values('A')`                 |

### Example 
```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print(df.head())
print("Mean age:", df['Age'].mean())
```

---

## Scikit-learn – Machine Learning in Python
**Purpose:** Implement ML algorithms and preprocessing in a consistent API.

**Why it’s important in ML:** It gives you a clean, unified way to train, evaluate, and deploy models.

**Key features:**
- Preprocessing (StandardScaler, OneHotEncoder, train_test_split)
- Built-in datasets (load_iris, load_digits, fetch_california_housing)
- Many algorithms (linear models, trees, clustering, etc.)
- Evaluation metrics (accuracy_score, mean_squared_error)
- Pipelines for chaining steps

| Category      | Common Functions / Classes                                         | Example                                                                     |
| ------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Datasets      | `load_iris`, `load_digits`, `train_test_split`                     | `from sklearn.datasets import load_iris`                                    |
| Preprocessing | `StandardScaler`, `OneHotEncoder`, `MinMaxScaler`                  | `scaler.fit_transform(X)`                                                   |
| Models        | `LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier` | `model.fit(X, y)`                                                           |
| Metrics       | `accuracy_score`, `mean_squared_error`, `classification_report`    | `accuracy_score(y_test, y_pred)`                                            |
| Pipelines     | `Pipeline`, `make_pipeline`                                        | `Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])` |


### Example
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
---
## Small note 
> It's not necessary to memorize every function now — just get familiar with which and where things live in each library. This cheat sheet will be a quick reference.