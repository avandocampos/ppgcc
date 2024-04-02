import os
import zipfile
import requests
import pandas as pd
import numpy as np
from classifiers import KNN
from sklearn.metrics import accuracy_score, confusion_matrix


# Função para carregar o conjunto de dados Iris do repositório UCI
def load_iris_uci():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    iris_data = pd.read_csv(url, names=columns)
    X = iris_data.iloc[:, :-1].values
    y = iris_data.iloc[:, -1].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).values

    return X, y


# Função para baixar e extrair o conjunto de dados da Coluna Vertebral do repositório UCI
def load_vertebral_column_uci():
    # URL do arquivo ZIP
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
    # Caminho local para salvar o arquivo ZIP
    zip_path = "vertebral_column_data.zip"
    # Caminho local para o arquivo de dados extraído
    data_path = "column_3C.dat"

    # Baixar o arquivo ZIP se ainda não foi baixado
    if not os.path.exists(zip_path):
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)

    # Extrair o arquivo de dados do ZIP se ainda não foi extraído
    if not os.path.exists(data_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()

    # Ler o arquivo de dados
    column_names = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'class']
    vertebral_data = pd.read_csv(data_path, header=None, sep=' ', names=column_names)
    X = vertebral_data.iloc[:, :-1].values
    y = vertebral_data.iloc[:, -1].replace({'DH': 0, 'SL': 1, 'NO': 2}).values

    return X, y


def generate_artificial_dataset():
    np.random.seed(42)

    # Parâmetros para a Classe 1
    mean1 = [1, 1]
    cov1 = [[0.1, 0], [0, 0.1]]
    class1 = np.random.multivariate_normal(mean1, cov1, 10)

    # Parâmetros para a Classe 0
    mean2 = [0, 0]
    cov2 = [[0.1, 0], [0, 0.1]]
    class0_1 = np.random.multivariate_normal(mean2, cov2, 10)

    mean3 = [0, 1]
    cov3 = [0.1, 0], [0, 0.1]
    class0_2 = np.random.multivariate_normal(mean3, cov3, 10)

    mean4 = [1, 0]
    cov4 = [[0.1, 0], [0, 0.1]]
    class0_3 = np.random.multivariate_normal(mean4, cov4, 10)

    class0 = np.vstack((class0_1, class0_2, class0_3))

    # Combinar as classes
    X_artificial = np.vstack((class1, class0))
    y_artificial = np.array([1]*10 + [0]*30)

    return X_artificial, y_artificial


# Função para dividir o conjunto de dados em treino e teste
def train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# Função para encontrar o número ótimo de vizinhos para KNN
def find_optimal_k(X_train, y_train, X_test, y_test, max_k=30):
    accuracies = []

    for k in range(1, max_k + 1):
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)

    optimal_k = np.argmax(accuracies) + 1

    return optimal_k, accuracies


def holdout_evaluation(X, y, classifier, num_trials=20, test_size=0.3, random_state=42):
    accuracies = []
    last_conf_matrix = None

    for i in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state+i)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        last_conf_matrix = confusion_matrix(y_test, y_pred)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return mean_accuracy, std_accuracy, last_conf_matrix
