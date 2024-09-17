import numpy as np
from scipy.stats import multivariate_normal

class GaussianDiscriminantAnalysis:
    def __init__(self, mode='LDA', reg_param=0.01):
        self.mode = mode  # 'LDA' para Discriminante Linear, 'QDA' para Quadrático
        self.reg_param = reg_param
        self.means = {}
        self.covariances = {}
        self.priors = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        overall_cov = np.zeros((X.shape[1], X.shape[1]))

        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]
            if self.mode == 'QDA':
                cov = np.cov(X_cls, rowvar=False) + self.reg_param * np.eye(X_cls.shape[1])
                self.covariances[cls] = cov
            else:
                cov = np.cov(X_cls, rowvar=False)
                overall_cov += cov * (X_cls.shape[0] - 1)  # Weighted sum for pooled covariance

        if self.mode == 'LDA':
            overall_cov /= (X.shape[0] - len(self.classes))  # Pooled covariance
            overall_cov += self.reg_param * np.eye(overall_cov.shape[0])
            for cls in self.classes:
                self.covariances[cls] = overall_cov

    def predict(self, X):
        predictions = []
        for x in X:
            probs = {}
            for cls in self.classes:
                mean = self.means[cls]
                cov = self.covariances[cls]
                prior = self.priors[cls]
                probability_density = multivariate_normal(mean, cov, allow_singular=True).pdf(x)
                probs[cls] = probability_density * prior
            predictions.append(max(probs, key=probs.get))
        return np.array(predictions)
    
    def print_covariances(self):
        for cls, cov in self.covariances.items():
            print(f"Matriz de Covariância para a Classe {cls}:")
            print(np.round(cov, decimals=2))
            print()
    
    def print_means(self):
        for cls, mean in self.means.items():
            print(f"Média para a Classe {cls}:")
            print(np.round(mean, decimals=2))
            print()
