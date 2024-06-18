import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from classifiers import build_cnn
from utils import (
    load_data,
    kfold_evaluation_cnn,
    evaluate_svm,
    evaluate_svm_kfold,
    evaluate_knn,
    evaluate_knn_kfold
)
from extractors import (
    extract_hog_features,
    extract_lbp_features,
    extract_pca_features
)


if __name__ == "__main__":
    # Carregar e verificar os dados
    X, y = load_data("ocr_car_numbers_rotulado.txt")
    print("Dimensões dos dados:", X.shape)
    print("Dimensões dos rótulos:", y.shape)

    # Redimensionar os dados para imagens 35x35
    X = X.reshape(-1, 35, 35)
    print("Dimensões das imagens:", X.shape)

    # Extrair características
    hog_features = extract_hog_features(X)
    print("Dimensões das características HOG:", hog_features.shape)

    lbp_features = extract_lbp_features(X)
    print("Dimensões das características LBP:", lbp_features.shape)

    pca_features = extract_pca_features(X)
    print("Dimensões das características PCA:", pca_features.shape)

    # Avaliar SVM
    evaluate_svm(hog_features, y, "HOG")
    evaluate_svm(lbp_features, y, "LBP")
    evaluate_svm(pca_features, y, "PCA")

    # Avaliar KNN
    evaluate_knn(hog_features, y, "HOG")
    evaluate_knn(lbp_features, y, "LBP")
    evaluate_knn(pca_features, y, "PCA")

    # Preparar os dados para a CNN
    X_cnn = X.reshape(-1, 35, 35, 1)
    y_cnn = to_categorical(y, num_classes=10)

    # Treinar e avaliar a CNN
    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)
    cnn = build_cnn(input_shape=(35, 35, 1))
    cnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    loss, acc = cnn.evaluate(X_test, y_test)
    y_pred = cnn.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm_cnn = confusion_matrix(y_true, y_pred_classes)
    print("Acurácia CNN:", acc)
    print("Matriz de Confusão CNN:\n", cm_cnn)

    # Definir classificadores
    svm_classifier = SVC(kernel='linear')
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Avaliar SVM com k-fold cross-validation
    evaluate_svm_kfold(hog_features, y, "HOG")
    evaluate_svm_kfold(lbp_features, y, "LBP")
    evaluate_svm_kfold(pca_features, y, "PCA")

    # Avaliar KNN com k-fold cross-validation
    evaluate_knn_kfold(hog_features, y, "HOG")
    evaluate_knn_kfold(lbp_features, y, "LBP")
    evaluate_knn_kfold(pca_features, y, "PCA")

    # Avaliar a CNN com k-fold cross-validation
    mean_acc_cnn, std_acc_cnn, mean_cm_cnn = kfold_evaluation_cnn(X_cnn, y_cnn, k=5, epochs=10)
    print("Acurácia Média CNN:", mean_acc_cnn)
    print("Desvio Padrão CNN:", std_acc_cnn)
    print("Matriz de Confusão Média CNN:\n", mean_cm_cnn)
