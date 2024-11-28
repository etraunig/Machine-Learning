import cudf as cudf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cuml.ensemble import RandomForestClassifier as cusRandomForestClassifier
from cuml.metrics import accuracy_score as cu_accuracy_score
from cuml.preprocessing.model_selection import train_test_split as cu_train_test_split

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score)

exibirMatrizes = False

# Base do código obtida do ChatGPT

# Datasets
data_treino = 'data/treino.csv'
data_teste = 'data/teste.csv'

# Carrega os dados em um DataFrame
dataTreino = pd.read_csv(data_treino)
dataTeste = pd.read_csv(data_teste)

# Convertendo variáveis Categóricas Ordinais
converter_variaveis = {
    'Education_Level': {
        'high school': 1,
        'bachelor': 2,
        'master': 3,
        'doctorate': 4
    }
}

dataTeste.replace(converter_variaveis, inplace=True)
dataTreino.replace(converter_variaveis, inplace=True)

# Lista de variáveis categóricas
colunas_categoria_treino = dataTreino.select_dtypes(include='object').columns.tolist()
colunas_categoria_teste = dataTeste.select_dtypes(include='object').columns.tolist()

# OneHotEncoding
dataTreino = pd.get_dummies(dataTreino, columns=colunas_categoria_treino, drop_first=True)
dataTeste = pd.get_dummies(dataTeste, columns=colunas_categoria_teste, drop_first=True)

# Variáveis independentes e alvo
X_treino = dataTreino.drop(['Preference'], axis=1)
y_treino = dataTreino['Preference']

X_teste = dataTeste.drop(['Preference'], axis=1)
y_teste = dataTeste['Preference']

#################### Funções ####################

# Função para avaliar o desempenho do classificador
# Base da função e prints obtidos do ChatGPT
def medidor_desempenho_classificador(modelo: object):
    lista_scores = []

    predicao_treino = modelo.predict(X_treino)
    predicao_teste = modelo.predict(X_teste)
    
    f1_treino= f1_score(y_treino, predicao_treino)
    f1_teste = f1_score(y_teste, predicao_teste)

    acur_treino = modelo.score(X_treino, y_treino)
    acur_teste = modelo.score(X_teste, y_teste)

    recall_treino = recall_score(y_treino, predicao_treino)
    recall_teste = recall_score(y_teste, predicao_teste)

    precisao_treino = precision_score(y_treino, predicao_treino)
    precisao_teste = precision_score(y_teste, predicao_teste)

    lista_scores.extend((acur_treino, acur_teste, recall_treino, recall_teste, precisao_treino, precisao_teste, f1_treino, f1_teste))

    print(f'\nAcurácia do Treino: {acur_treino}')
    print(f'Recall do Treino: {recall_treino}')
    print(f'Precisão do Treino: {precisao_treino}')
    print(f'F1-Score do Treino: {f1_treino}')
    print(f'\nAcurácia do Teste: {acur_teste}')
    print(f'Recall do Teste: {recall_teste}')
    print(f'Precisão do Teste: {precisao_teste}')
    print(f'F1-Score do Teste: {f1_teste}')
        
    return lista_scores

# Função para plotar a Matriz de Confusão
def plot_matriz_confusao(modelo: object, x: pd.DataFrame, y_atual: pd.Series):
    
    # Predição em Validação
    predicao_y = modelo.predict(x)

    # Pega os dados da Matriz de Confusão
    mat_conf = confusion_matrix(y_atual, predicao_y, labels=[0, 1])
    df_mat_conf = pd.DataFrame(mat_conf, index=['Não (0)', 'Sim (1)'], columns=['Não (0)', 'Sim (1)'])

    # List of labels for the Confusion Matrix
    contadores = [f'{valor:.0f}' for valor in mat_conf.flatten()]
    porcentagens = [f'{valor:.2f}%' for valor in (mat_conf.flatten()/np.sum(mat_conf))*100]

    labels = [f'{cont} - {porcent}' for cont, porcent in zip(contadores, porcentagens)]
    labels = np.asarray(labels).reshape(2, 2)

    # Plot the Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_mat_conf, annot=labels, fmt='')
    plt.xlabel('Classe Prevista', fontweight='bold')
    plt.ylabel('Classe Real', fontweight='bold')
    plt.show()

#################### Árvore de Decisão ####################
print("\nÁrvore de Decisão:")

a_decisao = DecisionTreeClassifier(criterion="entropy", random_state=1)

# Treinamento do modelo
a_decisao.fit(X_treino, y_treino)

# Testando o modelo
a_decisao_scores = medidor_desempenho_classificador(a_decisao)

# Matrizes de Confusão
if exibirMatrizes:
    plot_matriz_confusao(a_decisao, X_treino, y_treino)
    plot_matriz_confusao(a_decisao, X_teste, y_teste)

#################### K-Nearest Neighbors ####################
print("\nK-Nearest Neighbors:")

knn = KNeighborsClassifier()

# Grade de parâmetros para combinar
parametros = {'n_neighbors': np.arange(5, 10),
              'metric': ['euclidean', 'manhattan'],
              'weights': ['uniform', 'distance'],
              'algorithm': ['ball_tree', 'kd_tree']
             }

# Métrica usada para comparar as combinações de parâmetros
scorer_comp = metrics.make_scorer(metrics.f1_score)

# Roda a Grid Search
grid_search = GridSearchCV(knn, parametros, scoring=scorer_comp, cv=5)
grid_search = grid_search.fit(X_treino,y_treino)

# Cria o modelo com a melhor combinação
k_near = grid_search.best_estimator_

# Treina o modelo
k_near.fit(X_treino,y_treino)

# Resultados
k_near_score = medidor_desempenho_classificador(k_near)

if exibirMatrizes:
    plot_matriz_confusao(k_near, X_treino, y_treino)
    plot_matriz_confusao(k_near, X_teste, y_teste)
    
#################### Random Forest ####################
print("\nRandom Forest:")

randomForest = RandomForestClassifier(n_estimators=100, random_state=1)

# Parâmetros para combinar
parametros = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10,15,None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', None]
}

# Métrica para comparar as combinações
scorer_comp = metrics.make_scorer(metrics.f1_score)

# Roda a Grid Search
grid_search = GridSearchCV(randomForest, parametros, scoring=scorer_comp, cv=5)
grid_search = grid_search.fit(X_treino,y_treino)

# Cria o modelo com melhor combinação
random_forest = grid_search.best_estimator_

# Treina o modelo
random_forest.fit(X_treino,y_treino)

# Resultados
random_forest_scores = medidor_desempenho_classificador(random_forest)

# Matrizes de Confusão
if exibirMatrizes:
    plot_matriz_confusao(random_forest, X_treino, y_treino)
    plot_matriz_confusao(random_forest, X_teste, y_teste)
