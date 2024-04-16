import numpy as np


class GaussianBayesianClassifier:
    def __init__(self):
        self.means = {}
        self.covariances = {}
        self.priors = {}
        self.classes = []

    def calculate_mean(self, X):
        return np.mean(X, axis=0)

    def calculate_covariance(self, X, regularization_term=1e-6):
        covariance = np.cov(X, rowvar=False)
        
        return covariance + np.eye(covariance.shape[0]) * regularization_term

    def posterior_probability(self, x, mean, covariance, prior):
        dimension = len(mean)
        cov_inverse = np.linalg.inv(covariance)
        difference = x - mean
        exponent = -0.5 * np.dot(np.dot(difference.T, cov_inverse), difference)
        denominator = np.sqrt((2 * np.pi) ** dimension * np.linalg.det(covariance))
        
        return prior * np.exp(exponent) / denominator

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = self.calculate_mean(X_cls)
            self.covariances[cls] = self.calculate_covariance(X_cls)
            self.priors[cls] = len(X_cls) / len(X)

    def predict(self, X):
        predictions = []
        for x in X:
            probs = {cls: self.posterior_probability(x, self.means[cls], self.covariances[cls], self.priors[cls]) for cls in self.classes}
            predictions.append(max(probs, key=probs.get))
        
        return np.array(predictions)
