
import numpy as np

class KMeansClassifier:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X, y=None):
        np.random.seed(42)
        random_idx = np.random.permutation(X.shape[0])
        self.centroids = X[random_idx[:self.n_clusters]]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def print_covariances(self, X):
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            covariance = np.cov(cluster_points, rowvar=False)
            print(f"Covariance matrix for cluster {i}:{covariance}")

    def print_means(self):
        for i, centroid in enumerate(self.centroids):
            print(f"Centroid {i}: {centroid}")
