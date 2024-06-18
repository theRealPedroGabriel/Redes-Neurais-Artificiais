import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada da função sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Função para o Forward pass
def forward_pass(X, weights):
    inputs = []
    activations = [X]
    for W in weights:
        input = np.dot(activations[-1], W.T)
        activation = sigmoid(input)
        inputs.append(input)
        activations.append(np.hstack([np.ones((activation.shape[0], 1)), activation]))  # Add bias
    return inputs, activations


# Função para o Backward pass
def backward_pass(y, inputs, activations, weights, eta):
    deltas = [None] * len(weights)
    deltas[-1] = sigmoid_derivative(activations[-1][:, 1:]) * (y - activations[-1][:, 1:])  # Remove bias
    for l in range(len(deltas) - 2, -1, -1):
        Wb = weights[l + 1][:, 1:]  # Remove bias from weights
        deltas[l] = sigmoid_derivative(activations[l + 1][:, 1:]) * np.dot(deltas[l + 1], Wb)
    for l in range(len(weights)):
        weights[l] += eta * np.dot(deltas[l].T, activations[l])
    return weights


# Função para calcular o erro quadrático médio
def calculate_eqm(X, y, weights):
    _, activations = forward_pass(X, weights)
    predictions = activations[-1][:, 1:]  # Remove bias
    eqm = np.mean((y - predictions) ** 2) / 2
    return eqm


# Função para treinamento da MLP
def train_mlp(X, y, weights, eta, max_epochs, epsilon):
    for epoch in range(max_epochs):
        previous_eqm = calculate_eqm(X, y, weights)
        for i in range(X.shape[0]):
            inputs, activations = forward_pass(X[i].reshape(1, -1), weights)
            weights = backward_pass(y[i].reshape(1, -1), inputs, activations, weights, eta)
        current_eqm = calculate_eqm(X, y, weights)
        if abs(current_eqm - previous_eqm) < epsilon:
            break
    return weights


# Função para teste da MLP
def test_mlp(X, weights):
    _, activations = forward_pass(X, weights)
    return np.argmax(activations[-1][:, 1:], axis=1)  # Remove bias


# Carregar os dados
data = pd.read_csv(r'C:\Users\pgsmc\PycharmProjects\pythonProject5\aerogerador (3) (1).dat', delimiter=r'\s+', header=None)

# Verificar e limpar os dados
data = data.dropna()

# Dividir os dados em características (X) e rótulos (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Assumindo que y é uma variável contínua, vamos discretizá-la para transformá-la em um problema de classificação
# (por exemplo, converter em 2 classes: valores menores que a mediana são 0, maiores ou iguais à mediana são 1)
median_value = np.median(y)
y = np.where(y < median_value, 0, 1)

# Transformar y em codificação one-hot
y_onehot = np.zeros((y.size, 2))
y_onehot[np.arange(y.size), y] = 1

# Parâmetros da MLP
input_dim = X.shape[1]  # Número de características
hidden_dim = 10  # Número de neurônios na camada escondida
output_dim = 2  # Número de classes
eta = 0.01  # Taxa de aprendizagem
max_epochs = 1000  # Número máximo de épocas
epsilon = 1e-5  # Critério de parada

# Inicializa os pesos da MLP
np.random.seed(42)
weights = [
    np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim + 1)),
    np.random.uniform(-0.5, 0.5, (output_dim, hidden_dim + 1))
]

# Realiza as 100 rodadas de treinamento e teste
accuracies = []
specificities = []
sensitivities = []
weights_history = []  # List to store weights for each round

for _ in range(10):
    # Divisão aleatória dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)
    y_train_onehot = np.zeros((y_train.size, 2))
    y_train_onehot[np.arange(y_train.size), y_train] = 1

    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Adicionar o bias às entradas
    X_train_scaled = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
    X_test_scaled = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

    # Treinar a MLP
    trained_weights = train_mlp(X_train_scaled, y_train_onehot, weights, eta, max_epochs, epsilon)
    weights_history.append(trained_weights.copy())  # Save a copy of weights
    # Testar a MLP
    predictions = test_mlp(X_test_scaled, trained_weights)

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