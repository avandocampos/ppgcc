# Reconhecimento de Padrões - Classificadores KNN e DMC

Este projeto consiste na implementação e avaliação dos classificadores K-Nearest Neighbors (KNN) e Distância Mínima ao Centróide (DMC) em três conjuntos de dados distintos: Iris, Coluna Vertebral e dados gerados artificialmente.

## Requisitos

Para executar o script, você precisa ter o Python instalado em sua máquina, juntamente com as seguintes bibliotecas:

    NumPy
    Matplotlib
    Pandas
    Requests
    Scikit-learn

Você pode instalar todas as dependências necessárias usando o arquivo requirements.txt incluído neste repositório:

    pip install -r requirements.txt

## Execução

Para executar o script principal, navegue até o diretório do projeto e execute o seguinte comando no terminal:

    python app.py

Isso iniciará o script, que irá treinar e avaliar os classificadores nos conjuntos de dados especificados, e gerar gráficos das superfícies de decisão.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

- classifiers.py: Contém a implementação dos classificadores KNN e DMC.
- utils.py: Contém funções auxiliares para carregar os conjuntos de dados, dividir os dados em conjuntos de treinamento e teste, e avaliar os classificadores.
- app.py: Script principal que executa a avaliação dos classificadores.
- requirements.txt: Lista de dependências necessárias para executar o projeto.
- README.md: Este arquivo, que fornece uma visão geral do projeto e instruções de execução.

## Autores

- Avando José e L. Campos

## Licença

Este projeto está licenciado sob a Licença Apache 2.0. Consulte o arquivo LICENSE para obter mais detalhes.
