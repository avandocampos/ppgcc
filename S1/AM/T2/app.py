import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
from utils import (
    load_iris_uci,
    load_vertebral_column_uci,
    load_breast_cancer_uci,
    load_dermatology_uci,
    generate_artificial_dataset,
    holdout_evaluation,
)
from classifier import GaussianBayesianClassifier


def evaluate_dataset(dataset_loader, dataset_name):
    X, y = dataset_loader()
    gnb = GaussianBayesianClassifier()
    mean_accuracy, std_accuracy, conf_matrix = holdout_evaluation(X, y, gnb)
    print(f"{dataset_name} - Mean accuracy: {mean_accuracy}, Standard deviation: {std_accuracy}")
    print("Confusion Matrix (last trial):")
    print(conf_matrix)


def plot_decision_surface(
        dataset_loader, feature_indices=(0, 1),
        feature_names=["Feature 1", "Feature 2"],
        class_names=[],
        resolution=0.02,
        dataset_name="Dataset"):
    # Carregar dados
    X, y = dataset_loader()
    X = X[:, list(feature_indices)]  # Seleciona apenas as duas características para visualização

    # Inicializar e treinar o classificador
    classifier = GaussianBayesianClassifier()
    classifier.fit(X, y)

    # Configurar marcadores e cores do mapa
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'purple', 'orange')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plotar a superfície de decisão
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)

    # Plotar amostras de classe
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)],
                    marker=markers[idx], label=class_names[cl])

    plt.title(f'Superfície de decisão para {dataset_name.split(sep=" ")[0]} usando o classificador Gaussiano Bayesiano')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.savefig(f"{dataset_name.lower().split(sep=' ')[0]}_decision_surface.png")


def plot_gaussians(dataset_loader, dataset_name=None):
    X, y = dataset_loader()
    labels = np.unique(y)
    print(f"Total data points: {X.shape[0]}")
    print(f"Unique labels: {labels}")
    print(f"Labels array: {y}")  # Debug: imprimir o array de labels

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', 'x', '^', 's', 'D', 'v', '*']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for i, label in enumerate(labels):
        mask = (y == label)
        print(f"Class {label} mask size: {mask.size}, number of True elements: {np.sum(mask)}")  # Debug
        if np.sum(mask) == 0:
            continue  # Se não há elementos para uma classe, pular
        X_class = X[mask, :][:, [0, 1]]
        mean_class = np.mean(X_class, axis=0)
        cov_class = np.cov(X_class, rowvar=False)
        
        rv = multivariate_normal(mean=mean_class, cov=cov_class)
        _x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        _y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
        X_mesh, Y_mesh = np.meshgrid(_x, _y)
        Z = rv.pdf(np.dstack((X_mesh, Y_mesh)))
        
        ax.plot_surface(X_mesh, Y_mesh, Z, color=colors[i], alpha=0.5)
        ax.scatter(mean_class[0], mean_class[1], Z.max(), c=colors[i], marker=markers[i], s=100)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Densidade de Probabilidade')
    plt.savefig(f'{dataset_name}_gaussian.png')


def plot_gaussians_2d(dataset_loader, feature_indices=(0, 1)):
    X, y = dataset_loader()
    labels = np.unique(y)
    print(f"Total data points: {X.shape[0]}")
    print(f"Unique labels: {labels}")  # Confirmação das labels únicas identificadas
    print(f"Labels array: {y}")  # Saída completa do array de labels para verificação

    fig, ax = plt.subplots()

    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
    for i, label in enumerate(labels):
        mask = np.where(y == label)[0]
        print(f"Class {label}: mask size = {len(mask)}")  # Verificar o tamanho da máscara para cada classe

        if len(mask) == 0:
            print(f"No data points for class {label}. Skipping.")
            continue

        X_class = X[mask][:, feature_indices]
        mean_class = np.mean(X_class, axis=0)
        cov_class = np.cov(X_class, rowvar=False)
        rv = multivariate_normal(mean=mean_class, cov=cov_class)

        _x = np.linspace(np.min(X[:, feature_indices[0]]), np.max(X[:, feature_indices[0]]), 100)
        _y = np.linspace(np.min(X[:, feature_indices[1]]), np.max(X[:, feature_indices[1]]), 100)
        X_mesh, Y_mesh = np.meshgrid(_x, _y)
        pos = np.dstack((X_mesh, Y_mesh))
        Z = rv.pdf(pos)

        ax.contour(X_mesh, Y_mesh, Z, levels=5, colors=colors[i])
        ax.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], label=f'Class {label}')

    ax.set_title('Gaussian Distributions for Each Class')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    plt.savefig("gaussian_2d.png")


if __name__ == "__main__":

    """evaluate_dataset(load_iris_uci, "Iris Dataset")
    evaluate_dataset(load_vertebral_column_uci, "Vertebral Column Dataset")
    evaluate_dataset(load_breast_cancer_uci, "Breast Cancer Dataset")
    evaluate_dataset(load_dermatology_uci, "Dermatology Dataset")
    evaluate_dataset(generate_artificial_dataset, "Artificial I Dataset")

    plot_decision_surface(
        load_iris_uci,
        feature_indices=(0, 1),
        feature_names=["Comprimento da Sépala [cm]", "Largura da Sépala [cm]"],
        class_names=["Setosa", "Versicolor", "Virginica"],
        dataset_name="Iris Dataset"
    )

    plot_decision_surface(
        load_vertebral_column_uci,
        feature_indices=(0, 1),
        feature_names=["Pelvic Incident", "Pelvic Tilt"],
        class_names=["DH", "SL", "NO"],
        dataset_name="Vertebral Column Dataset"
    ) """

    """ plot_decision_surface(
        load_breast_cancer_uci,
        feature_indices=(0, 1),
        feature_names=["Raio Médio", "Textura Média"],
        class_names=["Maligno", "Benigno"],
        dataset_name="Breast Cancer Dataset"
    ) """

    """ plot_decision_surface(
        load_dermatology_uci,
        feature_indices=(0, 1),
        feature_names=["Feature 1", "Feature 2"],
        class_names=[
            "Dermatofitose",
            "Pitiríase rubra pilar",
            "Pitiríase líquenóide",
            "Pityriasis rosea",
            "Dermatite seborreica",
            "Psoríase"
        ],
        dataset_name="Dermatology Dataset"
    ) """
    """
    plot_decision_surface(
        generate_artificial_dataset,
        feature_indices=(0, 1),
        feature_names=["Feature 1", "Feature 2"],
        class_names=["1", "2", "3", "4"],
        dataset_name="Artificial I Dataset"
    )
 """
    """ plot_gaussians(dataset_loader=load_iris_uci, dataset_name="iris")
    plot_gaussians(dataset_loader=load_vertebral_column_uci, dataset_name="vertebral_column")
    plot_gaussians(dataset_loader=load_breast_cancer_uci, dataset_name="breast_cancer")
    plot_gaussians(dataset_loader=load_dermatology_uci, dataset_name="dermatology")
    plot_gaussians(dataset_loader=generate_artificial_dataset, dataset_name="artificial") """

    plot_gaussians_2d(load_breast_cancer_uci, (2,3))