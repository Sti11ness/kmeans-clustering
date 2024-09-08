# KMeans Clustering Algorithm with Visualization and KMeans++

## Project Overview

This project implements the KMeans clustering algorithm in Python with additional functionality for visualizing each step of the clustering process. It includes two initialization methods: random initialization and KMeans++ for more optimized centroid initialization. The implementation also supports the Manhattan and Euclidean distance metrics.

### Key Features
- **Customizable distance metrics**: Supports both Euclidean and Manhattan distance metrics.
- **KMeans++**: Provides better initialization of centroids for faster convergence.
- **Visualization**: Visualizes the clustering process and how centroids evolve with each iteration.
- **Hyperparameter tuning**: Allows easy tuning of parameters such as `k`, `tol`, `max_iter`, `method`, and `metric`.
- **Model evaluation**: Uses inertia and Rand Index to evaluate the quality of the clustering.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [KMeans++](#kmeans)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Visualization](#visualization)
- [Evaluation Metrics](#evaluation-metrics)
- [Examples](#examples)
- [References](#references)

## Installation
To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/Sti11ness/kmeans-clustering.git
cd kmeans-clustering
```

## Usage

The project consists of a `KMeans` class that allows for flexible clustering based on various parameters like distance metrics and initialization methods. The `fit` method is responsible for running the clustering algorithm, while the `evaluate` method calculates the inertia to assess model performance.

```python
from kmeans import KMeans
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')
# Also you can Normalize your data if you need(ofc you need)
# Initialize KMeans model
model = KMeans()
df_copy = df[['feature1', 'feature2']].copy(deep=True)
# Fit the model
model.fit(df_copy, k=3, method='kmeans++', metric='euclidean')

# Evaluate the model
score = model.evaluate(df)
```

### KMeans++

KMeans++ is a centroid initialization method that aims to speed up convergence by placing the initial centroids far apart. You can enable KMeans++ by specifying the method in the `fit` function:

```python
model.fit(df[['feature1', 'feature2']], k=3, method='kmeans++', metric='manhattan')
```

## Hyperparameter Tuning

To tune hyperparameters like the distance metric, initialization method, and the number of iterations, you can use a simple loop or a more sophisticated grid search approach:

```python
from itertools import product

params = {
    'metric': ['euclidean', 'manhattan'],
    'method': ['random', 'kmeans++'],
    'max_iter': [50, 100, 150],
    'tol': [1e-3, 1e-4]
}

best_score = -1
best_params = {}

# Grid search over hyperparameters
for metric, method, max_iter, tol in product(params['metric'], params['method'], params['max_iter'], params['tol']):
    model = KMeans()
    model.fit(df[['feature1', 'feature2']], k=3, method=method, metric=metric, max_iter=max_iter, tol=tol)
    score = model.evaluate(df)
    if score > best_score:
        best_score = score
        best_params = {'metric': metric, 'method': method, 'max_iter': max_iter, 'tol': tol}

print(f"Best Score: {best_score}")
print(f"Best Parameters: {best_params}")
```

## Visualization

The model saves the history of centroids and cluster assignments at each iteration, which allows for detailed visualization of the clustering process:

```python
import matplotlib.pyplot as plt

def visualize_kmeans(X, labels, centroids, iteration):
    plt.scatter(X['feature1'], X['feature2'], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x')
    plt.title(f"Iteration {iteration}")
    plt.show()

for i, (centroids, labels) in enumerate(zip(model.history['centroids'], model.history['clusters'])):
    visualize_kmeans(df[['feature1', 'feature2']], labels, centroids, i)
```

## Evaluation Metrics

- **Inertia**: Measures the sum of squared distances from each point to its assigned centroid. Lower inertia means better clustering.
- **Rand Index**: Measures the similarity between two data clusterings. It ranges from 0 to 1, where 1 represents perfect clustering.

Example:
```python
from sklearn.metrics import rand_score

rand_idx = rand_score(true_labels, predicted_labels)
print(f"Rand Index: {rand_idx}")
```

## Examples

Run the KMeans model with different settings and visualize the results:
```python
# Initialize model
model = KMeans()

# Fit model with k=3 and visualize
model.fit(df[['height', 'weight']], k=3, metric='euclidean', method='kmeans++', max_iter=100, track_history=True)

# Visualize clustering process
for i, (centroids, labels) in enumerate(zip(model.history['centroids'], model.history['clusters'])):
    visualize_kmeans(df[['height', 'weight']], labels, centroids, i)
```

## References

- [KMeans Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [KMeans++ Initialization](https://en.wikipedia.org/wiki/K-means%2B%2B)

