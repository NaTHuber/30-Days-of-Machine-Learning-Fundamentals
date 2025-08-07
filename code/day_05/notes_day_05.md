# Day 05. Data exploration with Pandas 
## Today's objetive 
To explore and understand the structure, distribution, and relationships in your dataset using Pandas. This step helps uncover patterns, spot anomalies, and generate hypotheses for modeling.

## What is EDA? 
Exploratory Data Analysis (EDA) is the process of understanding the data before applying any model. As we saw yesterday and the last few days, pandas provides powerful tools to:

- Inspect the dataset structure
- Summarize statistics
- Identify missing values
- Analyze distributions
- Detect potential outliers
- Spot correlations and patterns

a summary of main fuctions:

| Task                       | Function / Method          |
| -------------------------- | -------------------------- |
| View top rows              | `df.head()`                |
| View bottom rows           | `df.tail()`                |
| Summary stats              | `df.describe()`            |
| Data types                 | `df.dtypes`                |
| Dataset shape              | `df.shape`                 |
| Column names               | `df.columns`               |
| Count missing values       | `df.isna().sum()`          |
| Value counts (categorical) | `df['col'].value_counts()` |
| Correlation matrix         | `df.corr()`                |
| Unique values              | `df['col'].unique()`       |
| Group by + aggregate       | `df.groupby('col').mean()` |

## Small note 
> Exploration first, modeling later. Many modeling errors come from not understanding the quirks in the data 🌱