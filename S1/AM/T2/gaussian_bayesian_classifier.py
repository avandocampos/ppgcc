import numpy as np


def calculate_mean(X):
    return np.mean(X, axis=0)


def calculate_covariance(X, regularization_term=1e-6):
    covariance = np.cov(X, rowvar=False)

    return covariance + np.eye(covariance.shape[0]) * regularization_term


def posterior_probability(x, mean, covariance, prior):
    dimension = len(mean)
    cov_inverse = np.linalg.inv(covariance)
    difference = x - mean
    exponent = -0.5 * np.dot(np.dot(difference.T, cov_inverse), difference)
    denominator = np.sqrt((2 * np.pi) ** dimension * np.linalg.det(covariance))
    return prior * np.exp(exponent) / denominator


if __name__ == "__main__":
    # Example usage:
    X = np.array([[1, 2], [2, 3], [3, 4]])
    mean = calculate_mean(X)
    covariance = calculate_covariance(X)
    x = np.array([2, 3])
    prior = 1/3
    prob = posterior_probability(x, mean, covariance, prior)
    print(prob)
