# Day 04. Libraries: NumPy, Pandas, Scikit-learn (Deep Dive)
## Today's objetive 
Understand and practice NumPy, Pandas, and Scikit-learn, the core Python libraries for data manipulation and analysis. 

---

## NumPy – Numerical Python

**Purpose:** Fast numerical computation using arrays and matrices.

**Why it’s important in ML:** Almost every ML library (including Pandas and Scikit-learn) uses NumPy arrays under the hood.

**Key features:**
- N-dimensional arrays (ndarray)
- Broadcasting for arithmetic
- Linear algebra (np.dot, np.linalg.inv, etc.)
- Random number generation (np.random)
- Boolean masking and indexing

## Scikit-learn – Machine Learning in Python
**Purpose:** Implement ML algorithms and preprocessing in a consistent API.

**Why it’s important in ML:** It gives you a clean, unified way to train, evaluate, and deploy models.

**Key features:**
- Preprocessing (StandardScaler, OneHotEncoder, train_test_split)
- Built-in datasets (load_iris, load_digits, fetch_california_housing)
- Many algorithms (linear models, trees, clustering, etc.)
- Evaluation metrics (accuracy_score, mean_squared_error)
- Pipelines for chaining steps