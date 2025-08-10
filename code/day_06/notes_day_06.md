# Day 06. Data cleaning and handling missing values.
## Today's objective
Learn how to detect, handle, and clean missing or inconsistent data to ensure that datasets are reliable and ready for modeling. 

## Understanding this step 
Data cleaning is a crucial pre-processing step in any machine learning workflow.

Real-word datasets often contain:
- Missing values 
- Duplicated entries 
- Inconsistent data formats 
- Outliers

We can improves model accuracy and prevents bias handling these issues. 

## Common strategies for missing values

1. Deletion: removes rows or columns with missing values.
2. Imputation: replace missing values with a calculates value (mean, media, mode, constant or model-base prediction)
3. Flagging: Create an indicator column marking missing values 

## Example code using Pandas 
```python 
import pandas as pd 
import numpy as np 

# Example dataset 
data = {
    'Name':['Alice', 'Marcel', 'Natalia','David', None],
    'Age':[25,np.nan, 26, 22, 28],
    'Salary': [50000,42000, None, 58000, 60000],
}

df = pd.DataFrame(data)

print("Original DataFrame: ")
print(df)

# 1. Detecting missing values 
print("Missing values count: ")
print(df.isnull().sum())

# 2. Drop rows with missing values
df_drop = df.dropna()
print("\nAfter dropping rows:")
print(df_drop)

# 3. Fill missing values
df_fill = df.copy()
df_fill['Age'].fillna(df['Age'].mean(), inplace=True)
df_fill['Salary'].fillna(df['Salary'].median(), inplace=True)
df_fill['Name'].fillna("Unknown", inplace=True)

print("\nAfter filling missing values:")
print(df_fill)

```
We've already done this process in the past few days, but I hope it's now clearer why and how to do it.

## Small note 
> In machine learning, improper handling of missing values can introduce bias or cause your model to fail. A good practice is to keep a copy of the original data so you can revisit your cleaning decisions later.