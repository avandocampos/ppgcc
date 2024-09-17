import numpy as np
from scipy.spatial.distance import cdist

class ParzenBayesClassifier:
    def __init__(self, bandwidth=1.0):
        self.data_by_class = {}
        self.priors = None
        self.bandwidth = bandwidth  # Largura da janela

    def fit(self, X, y):
        # Identifica as classes únicas
        classes = np.unique(y)

        # Inicializa os parâmetros
        self.data_by_class = {}
        self.priors = {}

        for cls in classes:
            # Armazena os dados da classe
            X_cls = X[y == cls]
            self.data_by_class[cls] = X_cls
            self.priors[cls] = len(X_cls) / len(X)

    def _parzen_likelihood(self, X, X_cls):
        """
        Função de densidade baseada em Janela de Parzen
        """
        # Calcula a distância de cada ponto X para todos os pontos de X_cls
        distances = cdist(X, X_cls, metric='euclidean')

        # Função de kernel Gaussiano
        kernel_values = np.exp(-distances**2 / (2 * self.bandwidth**2))

        # Soma todas as contribuições dos pontos da classe
        likelihoods = np.sum(kernel_values, axis=1) / (X_cls.shape[0] * (self.bandwidth**X.shape[1]))
        
        return likelihoods

    def predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], len(self.data_by_class)))

        for i, cls in enumerate(self.data_by_class):
            X_cls = self.data_by_class[cls]
            prior = self.priors[cls]
            likelihood = self._parzen_likelihood(X, X_cls)
            probabilities[:, i] = likelihood * prior

        # Normalizar para obter probabilidades
        total_prob = np.sum(probabilities, axis=1, keepdims=True)
        probabilities /= total_prob
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def print_parzen_params(self):
        for cls, X_cls in self.data_by_class.items():
            print(f'Class {cls}:')
            print(f'Number of samples: {X_cls.shape[0]}')

    def write_parzen_params(self, file):
        file.write("Parzen Window Parameters:\n")
        for cls, X_cls in self.data_by_class.items():
            file.write(f'Class {cls}:\n')
            file.write(f'Number of samples: {X_cls.shape[0]}\n')
            file.write(f'Dimensionality: {X_cls.shape[1]}\n')
            file.write(f'Mean of samples: {np.mean(X_cls, axis=0)}\n')
            file.write(f'Variance of samples: {np.var(X_cls, axis=0)}\n')
            file.write(f'Bandwidth: {self.bandwidth}\n')
            file.write(f'Covariance matrix: {self.bandwidth**2 * np.eye(X_cls.shape[1])}\n')
            file.write("\n")
