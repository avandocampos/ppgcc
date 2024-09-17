import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
from utils import (
    load_iris_uci,
    load_vertebral_column_uci,
    generate_artificial_dataset,
    holdout_evaluation,
)
from classifier import KMeansClassifier


def plot_decision_surface(X, y, classifier, dataset_name, features=("Feature 0", "Feature 1")):
    h = .02  # Passo da malha

    # Criar a malha de pontos para plotar
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prever para cada ponto na malha
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plotar os pontos de treinamento
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=30, cmap=plt.cm.Paired)

    # Plotar os centróides
    centroids = classifier.centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='x')

    plt.title(f'Decision Surface of {dataset_name}')
    plt.xlabel(f"{features[0]}")
    plt.ylabel(f"{features[1]}")
    plt.show()


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

        print(f"Mean for class {label}: {mean_class}")
        print(f"Covariance for class {label}: {cov_class}")
        
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



def evaluate_dataset(dataset_name, dataset_loader, num_trials=20):
    X, y = dataset_loader()

    kmeans = KMeansClassifier(n_clusters=len(np.unique(y)))
    
    mean_accuracy, std_accuracy, conf_matrix = holdout_evaluation(X, y, kmeans, num_trials=num_trials)
    
    print(f"{dataset_name} - Mean accuracy: {mean_accuracy}, Standard deviation: {std_accuracy}")
    print("Confusion Matrix (last trial):")
    print(conf_matrix)

if __name__ == "__main__":

    plot_gaussians(load_iris_uci, "Iris Dataset")