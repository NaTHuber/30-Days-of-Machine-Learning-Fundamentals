# Day 11. Classification: K-Nearest Neighbors
## Today's objective 
Learn how the K-Nearest Neighbors (KNN) algorithm works for classification and apply it to a real dataset using Scikit-learn.
## Important concepts 

- **Idea:** KNN is a simple algorithm that classifies a data point based on the majority class among its k nearest neighbors.
- **Distance Metric:** Usually Euclidean distance, but others (like Manhattan, Minkowski) can be used.
- **Key parameter:**
    - k → number of neighbors considered.
    - Small k → model is sensitive to noise (overfitting).
    - Large k → smoother boundaries, but may underfit.
- **Pros:** Easy to understand, no training phase, works well on small datasets.
- **Cons:** Slow with large datasets, sensitive to irrelevant features, requires scaling of data.
