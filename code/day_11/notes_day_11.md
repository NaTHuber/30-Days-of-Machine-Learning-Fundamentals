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

## Example
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("KNN Confusion Matrix")
plt.show()

```

## Small note 
> KNN is widely used in recommendation systems, anomaly detection, and medical diagnosis because it makes decisions based on local similarity. However, in real-world projects, feature scaling and efficient distance computation (like KD-trees or Ball trees) become essential for large datasets 🌱