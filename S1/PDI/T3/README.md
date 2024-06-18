
# Image Classification and Feature Extraction

Este projeto implementa técnicas de extração de características e classificadores para reconhecimento óptico de caracteres (OCR).

## Estrutura do Projeto

- `classifiers.py`: Contém funções para treinar e avaliar classificadores SVM, KNN e um modelo CNN.
- `extractors.py`: Implementa funções para extração de características usando HOG, LBP e PCA.
- `main.py`: Script principal para carregar dados, extrair características, treinar e avaliar os classificadores.
- `utils.py`: Funções utilitárias para carregar dados, preprocessamento e avaliação.

## Requisitos

Os seguintes pacotes Python são necessários para executar o projeto:

- numpy
- scikit-learn
- scikit-image
- tensorflow

Você pode instalar os requisitos usando o seguinte comando:

```bash
pip install -r requirements.txt
```

## Como Executar

Certifique-se de que todos os requisitos estão instalados.  
Execute o script principal `main.py` para treinar e avaliar os modelos de classificação.

```bash
python main.py
```

## Funções Principais

### `classifiers.py`

- `train_evaluate_svm(features, labels)`: Treina e avalia um classificador SVM.
- `train_evaluate_knn(features, labels, k=3)`: Treina e avalia um classificador KNN.
- `build_cnn(input_shape)`: Constrói e retorna um modelo CNN.

### `extractors.py`

- `extract_hog_features(images)`: Extrai características HOG das imagens.
- `extract_lbp_features(images)`: Extrai características LBP das imagens.
- `apply_pca(features, n_components=50)`: Aplica PCA às características extraídas.

### `utils.py`

- `load_data(file_path)`: Carrega dados do arquivo especificado.
- `preprocess_data(X)`: Preprocessa os dados para adequação aos modelos.
- `kfold_evaluation(classifier, features, labels, k=10)`: Avalia o classificador usando k-fold cross-validation.
- `kfold_evaluation_cnn(X, y, k=5, epochs=10, batch_size=32)`: Avalia a CNN usando k-fold cross-validation.

## Autor

Avando José de Lima Campos
