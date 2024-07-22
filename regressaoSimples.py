import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Gerando dados fictícios
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Dividindo os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Criando o modelo de Regressão Linear
model = LinearRegression()
model.fit(x_train, y_train)

# Fazendo previsões
y_pred = model.predict(x_test)

# Visualizando os resultados
plt.scatter(x_test, y_test, color='black', label='Dados Reais')
plt.plot(x_test, y_pred, color='blue', linewidth=3, label='Linha de Regressão')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regressão Linear Simples')
plt.legend()
plt.show()

# Mostrando os coeficientes
print(f"Coeficiente: {model.coef_}")
print(f"Intercepto: {model.intercept_}")
