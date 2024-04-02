import numpy as np
from collections import Counter


# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Classe para o classificador KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Calcular a distância entre x e todos os pontos no conjunto de treinamento
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Ordenar as distâncias e retornar os índices dos primeiros k vizinhos
        k_indices = np.argsort(distances)[:self.k]
        # Extrair os rótulos dos k vizinhos mais próximos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Obter o rótulo mais comum entre os k vizinhos
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Classe para o classificador DMC
class DMC:
    def fit(self, X, y):
        self.centroids = {}
        for class_ in np.unique(y):
            self.centroids[class_] = np.mean(X[y == class_], axis=0)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, centroid) for centroid in self.centroids.values()]
        return np.argmin(distances)
