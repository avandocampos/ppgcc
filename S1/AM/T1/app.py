import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from classifiers import KNN, DMC
from utils import (
    load_iris_uci,
    train_test_split,
    find_optimal_k,
    load_vertebral_column_uci,
    holdout_evaluation,
    generate_artificial_dataset
)


def plot_decision_surface(X, y, classifier, class_names, test_idx=None, resolution=0.02):
    # Configurar marcadores e cores do mapa
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'purple', 'orange')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plotar a superfície de decisão
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plotar amostras de classe
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)],
                    marker=markers[idx], label=class_names[cl])

    # Destacar amostras de teste
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='black',
                    alpha=1.0, linewidth=1, marker='*',
                    s=55, label='Conjunto de teste')


def plot_decision_surface_dmc(X, y, classifier, class_name, resolution=0.02):
    # Configurar marcadores e cores do mapa
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plotar a superfície de decisão
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plotar amostras de classe
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)],
                    marker=markers[idx], label=class_name[cl])

    # Plotar os centróides
    for class_, centroid in classifier.centroids.items():
        plt.scatter(*centroid, marker='*', s=200, c=[cmap(class_)], label=f'Centroide {class_name[class_]}')


if __name__ == '__main__':
    # Carregar o conjunto de dados Iris do UCI
    X, y = load_iris_uci()

    # Dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Encontrar o número ótimo de vizinhos para KNN
    optimal_k, accuracies = find_optimal_k(X_train, y_train, X_test, y_test)
    print(f"Número ótimo de vizinhos para KNN: {optimal_k}")

    # Plotar as acurácias para diferentes valores de k
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xlabel('Número de Vizinhos')
    plt.ylabel('Acurácia')
    plt.title('Acurácia do KNN em função do número de vizinhos')
    plt.show()

    # Instanciar e treinar o classificador KNN com o k ótimo
    knn = KNN(k=optimal_k)
    knn.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste com KNN
    y_pred_knn = knn.predict(X_test)

    # Calcular a acurácia do KNN
    accuracy_knn = np.mean(y_pred_knn == y_test)
    print(f"Acurácia do KNN: {accuracy_knn}")

    # Instanciar e treinar o classificador DMC
    dmc = DMC()
    dmc.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste com DMC
    y_pred_dmc = dmc.predict(X_test)

    # Calcular a acurácia do DMC
    accuracy_dmc = np.mean(y_pred_dmc == y_test)
    print(f"Acurácia do DMC: {accuracy_dmc}")

    # Avaliar o desempenho do KNN e do DMC no conjunto de dados Iris e apresentar a matriz de confusão
    mean_accuracy_knn_iris, std_accuracy_knn_iris, conf_matrix_knn_iris = holdout_evaluation(X, y, KNN(k=optimal_k))
    mean_accuracy_dmc_iris, std_accuracy_dmc_iris, conf_matrix_dmc_iris = holdout_evaluation(X, y, DMC())

    print(f"KNN (Iris) - Acurácia média: {mean_accuracy_knn_iris}, Desvio padrão: {std_accuracy_knn_iris}")
    print(f"Matriz de confusão (KNN - Iris):\n{conf_matrix_knn_iris}")

    print(f"DMC (Iris) - Acurácia média: {mean_accuracy_dmc_iris}, Desvio padrão: {std_accuracy_dmc_iris}")
    print(f"Matriz de confusão (DMC - Iris):\n{conf_matrix_dmc_iris}")

    # Plotagem da superfície de decisão para o conjunto de dados Iris
    X_plot = X[:, :2]
    y_plot = y
    X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(X_plot, y_plot, test_size=0.3, random_state=42)
    knn_plot = KNN(k=optimal_k)
    knn_plot.fit(X_train_plot, y_train_plot)
    plt.figure()
    class_names_iris = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'} # Definir os nomes das classes
    plot_decision_surface(X_plot, y_plot, classifier=knn_plot, class_names=class_names_iris, test_idx=range(105, 150))
    plt.xlabel('Comprimento da sépala [cm]')
    plt.ylabel('Largura da sépala [cm]')
    plt.legend(loc='upper left')
    plt.savefig('iris_knn_decision_surface.png')

    # Dividir os dados em treino e teste (usando apenas os dois primeiros atributos para a plotagem)
    X_train, X_test, y_train, y_test = train_test_split(X[:, :2], y, test_size=0.3, random_state=42)

    # Instanciar e treinar o classificador DMC
    dmc = DMC()
    dmc.fit(X_train, y_train)

    # Plotar a superfície de decisão
    plt.figure()
    plot_decision_surface_dmc(X[:, :2], y, classifier=dmc, class_name=class_names_iris)
    plt.xlabel('Comprimento da sépala [cm]')
    plt.ylabel('Largura da sépala [cm]')
    plt.legend(loc='upper left')
    plt.savefig('iris_dmc_decision_surface.png')

    # Carregar o conjunto de dados da Coluna Vertebral do UCI
    X_coluna, y_coluna = load_vertebral_column_uci()

    # Dividir o conjunto de dados em treino e teste
    X_train_coluna, X_test_coluna, y_train_coluna, y_test_coluna = train_test_split(X_coluna, y_coluna, test_size=0.3, random_state=42)

    # Treinar e testar o classificador KNN
    knn.fit(X_train_coluna, y_train_coluna)
    y_pred_knn_coluna = knn.predict(X_test_coluna)
    accuracy_knn_coluna = np.mean(y_pred_knn_coluna == y_test_coluna)
    print(f"Acurácia do KNN no conjunto de dados da Coluna Vertebral: {accuracy_knn_coluna}")

    # Treinar e testar o classificador DMC
    dmc.fit(X_train_coluna, y_train_coluna)
    y_pred_dmc_coluna = dmc.predict(X_test_coluna)
    accuracy_dmc_coluna = np.mean(y_pred_dmc_coluna == y_test_coluna)
    print(f"Acurácia do DMC no conjunto de dados da Coluna Vertebral: {accuracy_dmc_coluna}")

    # Avaliar o desempenho do KNN e do DMC no conjunto de dados da Coluna Vertebral
    mean_accuracy_knn_coluna, std_accuracy_knn_coluna, conf_matrix_knn_coluna = holdout_evaluation(X_coluna, y_coluna, KNN(k=optimal_k))
    mean_accuracy_dmc_coluna, std_accuracy_dmc_coluna, conf_matrix_dmc_coluna = holdout_evaluation(X_coluna, y_coluna, DMC())

    print(f"KNN (Coluna Vertebral) - Acurácia média: {mean_accuracy_knn_coluna}, Desvio padrão: {std_accuracy_knn_coluna}")
    print(f"Matriz de confusão (KNN - Coluna Vertebral):\n{conf_matrix_knn_coluna}")

    print(f"DMC (Coluna Vertebral) - Acurácia média: {mean_accuracy_dmc_coluna}, Desvio padrão: {std_accuracy_dmc_coluna}")
    print(f"Matriz de confusão (DMC - Coluna Vertebral):\n{conf_matrix_dmc_coluna}")

    # Dividir os dados em treino e teste (usando apenas os dois primeiros atributos para a plotagem)
    X_train_coluna, X_test_coluna, y_train_coluna, y_test_coluna = train_test_split(X_coluna[:, :2], y_coluna, test_size=0.3, random_state=42)

    # Instanciar e treinar o classificador KNN com o k ótimo
    optimal_k_coluna, _ = find_optimal_k(X_train_coluna, y_train_coluna, X_test_coluna, y_test_coluna)
    knn_coluna = KNN(k=int(optimal_k_coluna))
    knn_coluna.fit(X_train_coluna, y_train_coluna)

    # Plotar a superfície de decisão para o KNN
    plt.figure()
    class_names_coluna = {0: 'Hérnia de Disco', 1: 'Espondilolistese', 2: 'Normal'}
    plot_decision_surface(X_coluna[:, :2], y_coluna, classifier=knn_coluna, class_names=class_names_coluna, resolution=0.1, test_idx=range(217, 310))
    plt.xlabel('Pelvic Incidence')
    plt.ylabel('Pelvic Tilt')
    plt.legend(loc='upper left')
    plt.savefig('coluna_knn_decision_surface.png')

    # Instanciar e treinar o classificador DMC
    dmc_coluna = DMC()
    dmc_coluna.fit(X_train_coluna, y_train_coluna)

    # Plotar a superfície de decisão para o DMC
    plt.figure()
    class_names_coluna = {0: 'Hérnia de Disco', 1: 'Espondilolistese', 2: 'Normal'}
    plot_decision_surface_dmc(X_coluna[:, :2], y_coluna, classifier=dmc_coluna, class_name=class_names_coluna, resolution=0.1)
    plt.xlabel('Pelvic Incidence')
    plt.ylabel('Pelvic Tilt')
    plt.legend(loc='upper left')
    plt.savefig('coluna_dmc_decision_surface.png')

    # Carregar o conjunto de dados Artificial I
    X_artificial, y_artificial = generate_artificial_dataset()

    # Dividir os dados em treino e teste
    X_train_artificial, X_test_artificial, y_train_artificial, y_test_artificial = train_test_split(X_artificial, y_artificial, test_size=0.3, random_state=42)

    # Avaliar o desempenho do KNN e do DMC no conjunto de dados Artificial I
    mean_accuracy_knn_artificial, std_accuracy_knn_artificial, conf_matrix_knn_artificial = holdout_evaluation(X_artificial, y_artificial, KNN(k=optimal_k))
    mean_accuracy_dmc_artificial, std_accuracy_dmc_artificial, conf_matrix_dmc_artificial = holdout_evaluation(X_artificial, y_artificial, DMC())

    print(f"KNN (Artificial I) - Acurácia média: {mean_accuracy_knn_artificial}, Desvio padrão: {std_accuracy_knn_artificial}")
    print(f"Matriz de confusão (KNN - Artificial I):\n{conf_matrix_knn_artificial}")

    print(f"DMC (Artificial I) - Acurácia média: {mean_accuracy_dmc_artificial}, Desvio padrão: {std_accuracy_dmc_artificial}")
    print(f"Matriz de confusão (DMC - Artificial I):\n{conf_matrix_dmc_artificial}")

    # Plotagem da superfície de decisão para o conjunto de dados Artificial I
    X_train_artificial_plot, X_test_artificial_plot, y_train_artificial_plot, y_test_artificial_plot = train_test_split(X_artificial, y_artificial, test_size=0.3, random_state=42)
    knn_plot_artificial = KNN(k=optimal_k)
    knn_plot_artificial.fit(X_train_artificial_plot, y_train_artificial_plot)
    plt.figure()
    class_names_artificial = {0: '0', 1: '1'}
    plot_decision_surface(X_artificial, y_artificial, classifier=knn_plot_artificial, class_names=class_names_artificial, test_idx=range(28, 40))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.savefig('artificial_decision_surface.png')

    # Carregar os dados artificiais
    X_artificial, y_artificial = generate_artificial_dataset()

    # Dividir os dados em treino e teste
    X_train_artificial, X_test_artificial, y_train_artificial, y_test_artificial = train_test_split(X_artificial, y_artificial, test_size=0.3, random_state=42)

    # Instanciar e treinar o classificador DMC
    dmc_artificial = DMC()
    dmc_artificial.fit(X_train_artificial, y_train_artificial)

    # Plotar a superfície de decisão e os centróides
    plt.figure()
    class_names_artificial = {0: '0', 1: '1'}
    plot_decision_surface_dmc(X_artificial, y_artificial, classifier=dmc_artificial, class_name=class_names_artificial)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.savefig('artificial_dmc_decision_surface.png')
