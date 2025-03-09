# **🏠 Predição de Preços de Casas**

## 📌 **Descrição do Projeto**
Este projeto tem como objetivo prever os preços de casas com base em um conjunto de dados fornecido. Foram utilizados diferentes algoritmos de *Machine Learning* para encontrar o modelo que melhor se ajusta aos dados. O dataset passou por um processo de **limpeza**, **tratamento** e **análise exploratória** para garantir melhores resultados.

---

## ⚙ **Tecnologias Utilizadas**

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-Learn**
- **Matplotlib**
- **Seaborn**

---

## 📊 **Modelos Utilizados**

Foram testados três modelos de aprendizado de máquina:

- **Regressão Linear** → Modelo base para prever preços de casas.
- **Árvore de Decisão** → Algoritmo que cria regras de decisão para melhorar a previsão.
- **K-Nearest Neighbors (KNN)** → Utiliza a proximidade dos vizinhos para prever os preços.

---

## 📈 **Avaliação dos Modelos**

Os modelos foram avaliados utilizando as seguintes métricas:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

---

## 🔍 **Passo a Passo do Modelo**

### 1️⃣ **Coleta e Exploração dos Dados**
O dataset foi carregado e inspecionado para entender sua estrutura e identificar valores ausentes ou inconsistentes:

```python
import pandas as pd
import seaborn as sns

# Carregar os dados
df = pd.read_csv('dados_casas.csv')

# Exibir as primeiras linhas
df.head()
```

### 2️⃣ **Limpeza e Pré-processamento dos Dados**
Realizamos a limpeza dos dados, incluindo:

- **Tratamento de valores ausentes**
- **Normalização de variáveis**
- **Transformação de variáveis categóricas**

```python
# Remover valores nulos
df = df.dropna()

# Converter variáveis categóricas em numéricas
df = pd.get_dummies(df, drop_first=True)
```

### 3️⃣ **Divisão do Dataset**
Separamos os dados em conjuntos de treino e teste:

```python
from sklearn.model_selection import train_test_split

X = df.drop('preco', axis=1)
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4️⃣ **Treinamento do Modelo**
Utilizamos um modelo de **Regressão Linear** para a predição:

```python
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)
```

### 5️⃣ **Avaliação do Modelo**
Avaliamos o desempenho utilizando métricas como o **erro médio absoluto (MAE)** e o coeficiente **R²**:

```python
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'R²: {r2}')
```

## 📌 **Contato**
Se tiver dúvidas ou sugestões, fique à vontade para **abrir uma issue** ou **entrar em contato**! 😊
