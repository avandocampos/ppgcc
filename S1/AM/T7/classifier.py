import numpy as np
from sklearn.mixture import GaussianMixture

class GMMBayesClassifier:
    def __init__(self, n_components=1):
        self.gmms = {}
        self.priors = None
        self.n_components = n_components

    def fit(self, X, y):
        # Identifica as classes únicas
        classes = np.unique(y)

        # Inicializa os parâmetros
        self.gmms = {}
        self.priors = {}

        for cls in classes:
            X_cls = X[y == cls]
            gmm = GaussianMixture(n_components=self.n_components, covariance_type='full')
            gmm.fit(X_cls)
            self.gmms[cls] = gmm
            self.priors[cls] = len(X_cls) / len(X)

    def _gmm_likelihood(self, X, gmm):
        return np.exp(gmm.score_samples(X))

    def predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], len(self.gmms)))
        
        for i, cls in enumerate(self.gmms):
            gmm = self.gmms[cls]
            prior = self.priors[cls]
            likelihood = self._gmm_likelihood(X, gmm)
            probabilities[:, i] = likelihood * prior

        # Normalizar para obter probabilidades
        total_prob = np.sum(probabilities, axis=1, keepdims=True)
        probabilities /= total_prob
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def print_gmm_params(self):
        for cls, gmm in self.gmms.items():
            print(f'Class {cls}:')
            print(f'Weights: {gmm.weights_}')
            print(f'Means: {gmm.means_}')
            print(f'Covariances: {gmm.covariances_}')
    
    def write_gmm_params(self, file):
        file.write("GMM Parameters:\n")
        for cls, gmm in self.gmms.items():
            file.write(f'Class {cls}:\n')
            file.write(f'Weights: {gmm.weights_}\n')
            file.write(f'Means: {gmm.means_}\n')
            file.write(f'Covariances: {gmm.covariances_}\n')
            file.write("\n")
