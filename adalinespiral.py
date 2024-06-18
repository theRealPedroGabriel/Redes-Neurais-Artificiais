

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Função de ativação linear para ADALINE
def linear_activation(x):
    return x


# Inicializar os pesos com valores aleatórios
def initialize_weights(input_dim):
    return np.random.uniform(-0.5, 0.5, input_dim + 1)


# Função para o Forward pass
def forward_pass(X, weights):
    return np.dot(X, weights.T)


# Função para calcular o erro quadrático médio (EQM)
def calculate_eqm(X, y, weights):
    predictions = forward_pass(X, weights)
    errors = y - predictions
    eqm = np.mean(errors ** 2) / 2
    return eqm


# Função para treinamento do ADALINE
def train_adaline(X, y, weights, eta, max_epochs, epsilon):
    for epoch in range(max_epochs):
        previous_eqm = calculate_eqm(X, y, weights)
        for i in range(X.shape[0]):
            y_pred = forward_pass(X[i], weights)
            error = y[i] - y_pred
            weights += eta * error * X[i]
        current_eqm = calculate_eqm(X, y, weights)
        if abs(current_eqm - previous_eqm) < epsilon:
            break
    return weights


# Função para teste do ADALINE
def test_adaline(X, weights):
    y_pred = forward_pass(X, weights)
    return np.where(y_pred >= 0.5, 1, 0)  # Limite de decisão em 0.5


# Carregar os dados
column_names = ['x', 'y', 'label']
data = pd.read_csv(r'C:\Users\pgsmc\PycharmProjects\pythonProject5\spiral.csv', delimiter=',', names=column_names)

# Dividir os dados em características (X) e rótulos (y)
X = data[['x', 'y']].values
y = data['label'].values


# Assumindo que y é uma variável contínua, vamos discretizá-la para transformá-la em um problema de classificação
# (por exemplo, converter em 2 classes: valores menores que a mediana são 0, maiores ou iguais à mediana são 1)
median_value = np.median(y)
y = np.where(y < median_value, 0, 1)

# Parâmetros do ADALINE
eta = 0.01  # Taxa de aprendizagem
max_epochs = 1000  # Número máximo de épocas
epsilon = 1e-5  # Critério de parada

# Realiza as 100 rodadas de treinamento e teste
accuracies = []
specificities = []
sensitivities = []
weights_history = []  # List to store weights for each round
for _ in range(10):
    # Divisão aleatória dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)

    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Adicionar o bias às entradas
    X_train_scaled = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
    X_test_scaled = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

    # Inicializa os pesos do ADALINE
    weights = initialize_weights(X_train_scaled.shape[1] - 1)

    # Treinar o ADALINE
    trained_weights = train_adaline(X_train_scaled, y_train, weights, eta, max_epochs, epsilon)
    weights_history.append(trained_weights.copy())  # Save a copy of weights
    # Testar o ADALINE
    predictions = test_adaline(X_test_scaled, trained_weights)

    # Matrizes de Confusão
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    # Calcula medidas
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Armazena resultados
    accuracies.append(accuracy)
    specificities.append(specificity)
    sensitivities.append(sensitivity)

# Calcula estatísticas
mean_acc = np.mean(accuracies)
min_acc = np.min(accuracies)
max_acc = np.max(accuracies)
median_acc = np.median(accuracies)
std_acc = np.std(accuracies)

mean_spec = np.mean(specificities)
min_spec = np.min(specificities)
max_spec = np.max(specificities)
median_spec = np.median(specificities)
std_spec = np.std(specificities)

mean_sens = np.mean(sensitivities)
min_sens = np.min(sensitivities)
max_sens = np.max(sensitivities)
median_sens = np.median(sensitivities)
std_sens = np.std(sensitivities)

# Resultados
print(f"Accuracy:\n  Mean: {mean_acc}\n  Min: {min_acc}\n  Max: {max_acc}\n  Median: {median_acc}\n  Std: {std_acc}")
print(
    f"Specificity:\n  Mean: {mean_spec}\n  Min: {min_spec}\n  Max: {max_spec}\n  Median: {median_spec}\n  Std: {std_spec}")
print(
    f"Sensitivity:\n  Mean: {mean_sens}\n  Min: {min_sens}\n  Max: {max_sens}\n  Median: {median_sens}\n  Std: {std_sens}")
print(weights_history)