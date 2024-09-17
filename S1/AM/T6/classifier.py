import numpy as np

class GaussianBayesClassifier:
    def __init__(self, rejection_cost=0.1):
        self.means = None
        self.covariances = None
        self.priors = None
        self.rejection_cost = rejection_cost
    
    def fit(self, X, y):
        # Identifica as classes únicas
        classes = np.unique(y)
        n_features = X.shape[1]
        
        # Inicializa os parâmetros
        self.means = {}
        self.covariances = {}
        self.priors = {}
        
        for cls in classes:
            X_cls = X[y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.covariances[cls] = np.cov(X_cls, rowvar=False)
            self.priors[cls] = len(X_cls) / len(X)
    
    def _gaussian_likelihood(self, X, mean, covariance):
        n_features = X.shape[1]
        covariance_det = np.linalg.det(covariance)
        covariance_inv = np.linalg.inv(covariance)
        factor = 1 / np.sqrt((2 * np.pi) ** n_features * covariance_det)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ covariance_inv * diff, axis=1)
        return factor * np.exp(exponent)
    
    def predict(self, X):
        likelihoods = {}
        for cls in self.means:
            likelihoods[cls] = self.priors[cls] * self._gaussian_likelihood(X, self.means[cls], self.covariances[cls])
        
        # Calcula a evidência (normalização)
        total_likelihood = np.sum(list(likelihoods.values()), axis=0)
        
        # Calcula as probabilidades a posteriori
        posteriors = {cls: likelihood / total_likelihood for cls, likelihood in likelihoods.items()}
        
        # Aplica a opção de rejeição
        max_posteriors = np.max(list(posteriors.values()), axis=0)
        rejected = max_posteriors < (1 - self.rejection_cost)
        
        # Predição final com rejeição
        y_pred = np.full(X.shape[0], -1)  # Inicializa com -1 para as amostras rejeitadas
        for i in range(X.shape[0]):
            for cls, prob in posteriors.items():
                if prob[i] == max_posteriors[i]:
                    y_pred[i] = cls
        
        y_pred[rejected] = -1  # Marca as amostras rejeitadas com -1
        
        return y_pred
    
    def print_covariances(self):
        for cls, cov in self.covariances.items():
            print(f"Class {cls} covariance matrix:\n{cov}\n")
    
    def print_means(self):
        for cls, mean in self.means.items():
            print(f"Class {cls} mean vector:\n{mean}\n")
