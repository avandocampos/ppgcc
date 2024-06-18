from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


def load_data(file_path):
    from numpy import array
    
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(int, line.strip().split()))
            data.append(values[:-1])
            labels.append(values[-1])
    return array(data), array(labels)


def preprocess_data(X):
    return X.reshape(-1, 35, 35)


def kfold_evaluation(classifier, features, labels, k=10):
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    confusion_matrices = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    return mean_accuracy, std_accuracy, mean_confusion_matrix


def kfold_evaluation_cnn(X, y, k=5, epochs=10, batch_size=32):
    from classifiers import build_cnn
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    confusion_matrices = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = build_cnn(input_shape=(35, 35, 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
        
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracies.append(acc)
        confusion_matrices.append(confusion_matrix(y_true, y_pred_classes))

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    return mean_accuracy, std_accuracy, mean_confusion_matrix


def evaluate_knn_kfold(features, y, feature_name):
    mean_acc, std_acc, mean_cm = kfold_evaluation(knn_classifier, features, y)
    print(f"Acurácia Média KNN com {feature_name}:", mean_acc)
    print(f"Desvio Padrão KNN com {feature_name}:", std_acc)
    print(f"Matriz de Confusão Média KNN com {feature_name}:\n", mean_cm)


def evaluate_svm(features, y, feature_name):
    acc, cm = train_evaluate_svm(features, y)
    print(f"Acurácia SVM com {feature_name}:", acc)
    print(f"Matriz de Confusão SVM com {feature_name}:\n", cm)


def evaluate_svm_kfold(features, y, feature_name):
    mean_acc, std_acc, mean_cm = kfold_evaluation(svm_classifier, features, y)
    print(f"Acurácia Média SVM com {feature_name}:", mean_acc)
    print(f"Desvio Padrão SVM com {feature_name}:", std_acc)
    print(f"Matriz de Confusão Média SVM com {feature_name}:\n", mean_cm)


def evaluate_knn(features, y, feature_name):
    acc, cm = train_evaluate_knn(features, y)
    print(f"Acurácia KNN com {feature_name}:", acc)
    print(f"Matriz de Confusão KNN com {feature_name}:\n", cm)