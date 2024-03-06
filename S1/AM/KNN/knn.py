from ucimlrepo import fetch_ucirepo 
import numpy as np
from collections import Counter

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Função para dividir o conjunto de dados em conjuntos de treinamento e teste
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(y))
    test_set_size = int(len(y) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Implementação do k-NN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Calcular as distâncias entre x e todos os exemplos no conjunto de treinamento
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Ordenar por distância e retornar os índices dos primeiros k vizinhos
        k_indices = np.argsort(distances)[:self.k]
        # Extrair os rótulos dos k vizinhos mais próximos
        k_nearest_labels = self.y_train[k_indices]
        # Retornar o rótulo mais comum
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Carregar o conjunto de dados Iris
def load_iris_dataset():
    iris = fetch_ucirepo(id=53)
    X = iris.data.features 
    y = iris.data.targets 
    return X, y

# Carregar o conjunto de dados e dividir em conjuntos de treinamento e teste
X, y = load_iris_dataset()
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Inicializar e treinar o classificador k-NN
knn = KNN(k=3)
knn.fit(X_train, y_train.ravel())

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Calcular a precisão do modelo
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


import matplotlib.pyplot as plt
import matplotlib 

matplotlib.use('qtagg')

# Lista para armazenar as precisões para cada valor de k
accuracies = []

# Valores de k para testar
k_values = range(1, 10)

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train, y_train.ravel())
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Criar o gráfico
plt.plot(k_values, accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy for different values of k')
plt.show()