import os
import zipfile
import requests
import numpy as np
import pandas as pd


def load_iris_uci():
    # URL do arquivo de dados
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # Caminho local para salvar o arquivo de dados
    data_path = "iris.data"

    # Baixar o arquivo de dados se ainda não foi baixado
    if not os.path.exists(data_path):
        response = requests.get(url)
        with open(data_path, "w") as file:
            file.write(response.text)

    # Ler o arquivo de dados
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    iris_data = pd.read_csv(data_path, header=None, names=column_names)

    # Considerar apenas Setosa (0) vs Versicolor e Virginica (1)
    iris_data['class'] = iris_data['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 1})
    X = iris_data.iloc[:, :-1].values
    y = iris_data['class'].values

    # Verificações de integridade
    print(f"Shape of X: {X.shape}")  # Deve ser (150, 4)
    print(f"Shape of y: {y.shape}")  # Deve ser (150,)
    print(f"Unique classes in y: {np.unique(y)}")  # Deve ser [0, 1]

    return X, y


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
    
    # Considerar Normal (0) vs Hérnia de Disco e Espondilolistese (1)
    vertebral_data['class'] = vertebral_data['class'].replace({'NO': 0, 'DH': 1, 'SL': 1})
    X = vertebral_data.iloc[:, :-1].values
    y = vertebral_data['class'].values

    return X, y


def generate_artificial_dataset():
    np.random.seed(42)

    # Parameters for the regions of Class 0
    mean_0_0 = [0.2, 0.2]
    cov_0_0 = [[0.02, 0], [0, 0.02]]
    region_0_0 = np.random.multivariate_normal(mean_0_0, cov_0_0, 17)

    mean_1_0 = [0.8, 0.2]
    cov_1_0 = [[0.02, 0], [0, 0.02]]
    region_1_0 = np.random.multivariate_normal(mean_1_0, cov_1_0, 17)

    mean_0_1 = [0.2, 0.8]
    cov_0_1 = [[0.02, 0], [0, 0.02]]
    region_0_1 = np.random.multivariate_normal(mean_0_1, cov_0_1, 16)

    class0 = np.vstack((region_0_0, region_1_0, region_0_1))

    # Parameters for the regions of Class 1
    mean_class1 = [1.2, 1.2]
    cov_class1 = [[0.05, 0], [0, 0.05]]
    class1 = np.random.multivariate_normal(mean_class1, cov_class1, 50)

    X_artificial = np.vstack((class0, class1))
    y_artificial = np.array([0]*50 + [1]*50)

    return X_artificial, y_artificial

def train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def holdout_evaluation(X, y, classifier, num_trials=20, test_size=0.3, random_state=42):
    accuracies = []
    last_conf_matrix = None

    for i in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state+i)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        # Corrigido: Remover o argumento X_train
        classifier.print_covariances()
        classifier.print_means()
        
        # Se houver rejeição, remova ou marque as amostras rejeitadas antes de calcular a acurácia e a matriz de confusão
        valid_indices = y_pred != -1  # Supondo que o classificador retorne -1 para rejeição
        y_test = y_test[valid_indices]
        y_pred = y_pred[valid_indices]
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        last_conf_matrix = confusion_matrix(y_test, y_pred)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return mean_accuracy, std_accuracy, last_conf_matrix


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            conf_matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    return conf_matrix
