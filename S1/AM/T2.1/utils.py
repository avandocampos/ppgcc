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
    # Tratamento correto dos nomes de classe para evitar problemas de broadcasting
    iris_data['class'] = iris_data['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    X = iris_data.iloc[:, :-1].values
    y = iris_data['class'].values

    # Verificações de integridade
    print(f"Shape of X: {X.shape}")  # Deve ser (150, 4)
    print(f"Shape of y: {y.shape}")  # Deve ser (150,)
    print(f"Unique classes in y: {np.unique(y)}")  # Deve ser [0, 1, 2]

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
    X = vertebral_data.iloc[:, :-1].values
    y = vertebral_data.iloc[:, -1].replace({'DH': 0, 'SL': 1, 'NO': 2}).values

    return X, y


def load_breast_cancer_uci():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
    column_names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    data = pd.read_csv(url, header=None, names=column_names)
    data = data.replace('?', np.nan)
    data.dropna(inplace=True)

    # Converter atributos categóricos em numéricos
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category').cat.codes

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    return X, y


def load_dermatology_uci():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
    column_names = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules',
                    'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement',
                    'family_history', 'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'PNL_infiltrate',
                    'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
                    'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis',
                    'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of_the_granular_layer',
                    'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw_tooth_appearance_of_retes',
                    'follicular_horn_plug', 'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
                    'band_like_infiltrate', 'age', 'class']
    data = pd.read_csv(url, header=None, names=column_names, na_values="?")
    data.dropna(inplace=True)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y


def generate_artificial_dataset():
    np.random.seed(42)

    # Parâmetros para a Classe 1
    mean1 = [1, 1]
    cov1 = [[0.1, 0], [0, 0.1]]
    class1 = np.random.multivariate_normal(mean1, cov1, 10)

    # Parâmetros para a Classe 2
    mean2 = [0, 0]
    cov2 = [[0.1, 0], [0, 0.1]]
    class2 = np.random.multivariate_normal(mean2, cov2, 10)

    # Parâmetros para a Classe 3
    mean3 = [1, 0]
    cov3 = [[0.1, 0], [0, 0.1]]
    class3 = np.random.multivariate_normal(mean3, cov3, 10)

    # Combinar as classes
    X_artificial = np.vstack((class1, class2, class3))
    y_artificial = np.array([1]*10 + [2]*10 + [3]*10)

    return X_artificial, y_artificial


# Função para dividir o conjunto de dados em treino e teste
def train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            conf_matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    return conf_matrix


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
