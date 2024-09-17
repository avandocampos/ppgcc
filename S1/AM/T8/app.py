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
from classifier import ParzenBayesClassifier


def plot_decision_surface(load_data_func, feature_indices, feature_names, class_names, dataset_name, resolution=0.02):

    X, y = load_data_func()

    if max(feature_indices) >= X.shape[1]:
        raise ValueError("feature_indices contém um índice fora do intervalo das colunas de X")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo número de amostras")

    X = X[:, list(feature_indices)]

    classifier = ParzenBayesClassifier()
    classifier.fit(X, y)

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'purple', 'orange')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])  # Prevendo os pontos para a superfície
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)

    unique_classes, y_indices = np.unique(y, return_inverse=True)
    for idx, cl in enumerate(unique_classes):
        plt.scatter(x=X[y_indices == idx, 0], y=X[y_indices == idx, 1],
                    alpha=0.8, c=[cmap(idx)],
                    marker=markers[idx], label=class_names[idx])

    plt.title(f'Superfície de decisão para {dataset_name} usando Janela de Parzen')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.savefig(f"{dataset_name.lower().split(sep=' ')[0]}_decision_surface.png")
    plt.show()


def evaluate_dataset(dataset_name, dataset_loader, num_trials=20):

    X, y = dataset_loader()
    
    classifier = ParzenBayesClassifier()
    
    filename = f"results_{dataset_name.lower().replace(' ', '_')}.txt"

    with open(filename, 'w') as f:
        f.write(f"Results for {dataset_name}:\n")
        
        mean_accuracy, std_accuracy, mean_conf_matrix = holdout_evaluation(X, y, classifier, num_trials=num_trials, results_file=f)
        
        f.write(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write("Mean Confusion Matrix:\n")
        f.write(str(mean_conf_matrix))
    
    return mean_accuracy, std_accuracy, mean_conf_matrix


if __name__ == "__main__":

    plot_decision_surface(generate_artificial_dataset, [0, 1], ['Feature 0', 'Feature 1'], ['Class 0', 'Class'], 'Artificial Dataset')