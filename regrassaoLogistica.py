import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carregando dados fictícios
data = {
    'idade': [25, 34, 45, 52, 23, 43, 35, 31, 41, 50],
    'renda': [40000, 60000, 80000, 120000, 30000, 70000, 90000, 50000, 75000, 110000],
    'escolaridade': [12, 16, 14, 18, 10, 16, 15, 13, 16, 17],
    'ganhou': [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Separando as características (X) e o alvo (y)
X = df[['idade', 'renda', 'escolaridade']]
y = df['ganhou']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de Regressão Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Acurácia: {accuracy}")
print("Matriz de Confusão:")
print(conf_matrix)
print("Relatório de Classificação:")
print(class_report)
