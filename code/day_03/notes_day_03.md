# Day 03. Data collection and preparation.
## Today's objective
Understand the process of collecting and preparing data for machine learning, ensuring it is clean, relevant, and ready for modeling.

---

## Important concepts for the process
- **Data Collection:** Gathering data from different sources such as CSV/Excel files, databases, APIs, web scraping, or public datasets.

- **Data Preparation:** Transforming raw data into a suitable format:
    - Cleaning (handling missing/duplicate values)
    - Type conversion (e.g., string to numeric)
    - Formatting (dates, categories, etc.)
    - Initial feature selection

```mermaid 
flowchart TD
    A[Start] --> B[Data Collection]
    B --> B1[From files: CSV, Excel, JSON]
    B --> B2[From databases: SQL, NoSQL]
    B --> B3[From APIs or Web Scraping]
    B --> B4[From public datasets: Kaggle, UCI, etc.]
    B1 --> C[Data Inspection]
    B2 --> C
    B3 --> C
    B4 --> C

    C --> D[Data Cleaning]
    D --> D1[Handle missing values]
    D --> D2[Remove duplicates]
    D --> D3[Fix inconsistent formats] 

    D --> E[Data Transformation]
    E --> E1[Convert data types]
    E --> E2[Encode categorical variables]
    E --> E3[Scale/normalize numerical features]

    E --> F[Feature Selection]
    F --> F1[Remove irrelevant columns]
    F --> F2[Create new derived features]

    F --> G[Save prepared dataset]
    G --> H[Ready for ML Modeling]

```
---

## Small note 
> Data preparation often takes 80% of the total ML project time. Don’t rush it, better data means better models 🌱