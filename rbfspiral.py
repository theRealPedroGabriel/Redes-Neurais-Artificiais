import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Funções de ativação e suas derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Função de ativação radial gaussiana
def rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * s ** 2))


# Inicializa centros
def initialize_centers(X, q):
    return X[np.random.choice(X.shape[0], q, replace=False)]


# Fase de ajuste dos centros
def adjust_centers(X, q, eta, max_iterations):
    centers = initialize_centers(X, q)
    for t in range(max_iterations):
        x = X[np.random.choice(X.shape[0])]
        i_star = np.argmin(np.linalg.norm(x - centers, axis=1))
        centers[i_star] += eta * (x - centers[i_star])
    return centers


# Função para calcular as ativações RBF
def calculate_rbf_activations(X, centers, s):
    activations = np.zeros((X.shape[0], centers.shape[0]))
    for i in range(X.shape[0]):
        for j in range(centers.shape[0]):
            activations[i, j] = rbf(X[i], centers[j], s)
    return activations


# Inicializa pesos
def initialize_weights(input_dim, output_dim):
    return np.random.uniform(-0.5, 0.5, (output_dim, input_dim + 1))


# Função para o Forward pass
def forward_pass(X, weights):
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    return sigmoid(np.dot(X_bias, weights.T))


# Função para treinamento da RBF
def train_rbf(X, y, centers, eta, max_epochs, s):
    activations = calculate_rbf_activations(X, centers, s)
    weights = initialize_weights(activations.shape[1], y.shape[1])
    for epoch in range(max_epochs):
        for i in range(X.shape[0]):
            y_pred = forward_pass(activations[i].reshape(1, -1), weights)
            error = y[i] - y_pred
            weights += eta * error.T.dot(np.insert(activations[i], 0, 1).reshape(1, -1))
    return weights


# Função para teste da RBF
def test_rbf(X, centers, weights, s):
    activations = calculate_rbf_activations(X, centers, s)
    y_pred = forward_pass(activations, weights)
    return np.argmax(y_pred, axis=1)


# Carregar os dados
column_names = ['x', 'y', 'label']
data = pd.read_csv(r'C:\Users\pgsmc\PycharmProjects\pythonProject5\spiral.csv', delimiter=',', names=column_names)

# Dividir os dados em características (X) e rótulos (y)
X = data[['x', 'y']].values
y = data['label'].values

# Verificar os valores únicos em y e convertê-los para inteiros
unique_labels = np.unique(y)
label_map = {unique_labels[i]: i for i in range(len(unique_labels))}
y = np.vectorize(label_map.get)(y)
y_onehot = np.zeros((y.size, len(unique_labels)))
y_onehot[np.arange(y.size), y] = 1

# Parâmetros da RBF
q = 10  # Número de funções de base radial
eta = 0.01  # Taxa de aprendizagem
max_iterations = 100  # Número máximo de iterações para ajustar os centros
max_epochs = 1000  # Número máximo de épocas para treinamento dos pesos
s = 1.0  # Largura das funções de base radial

# Realiza as 100 rodadas de treinamento e teste
accuracies = []
specificities = []
sensitivities = []
weights_history = []  # List to store weights for each round

for _ in range(10):
    # Divisão aleatória dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)
    y_train_onehot = np.zeros((y_train.size, len(unique_labels)))
    y_train_onehot[np.arange(y_train.size), y_train] = 1

    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ajustar os centros
    centers = adjust_centers(X_train_scaled, q, eta, max_iterations)

    # Treinar a RBF
    weights = train_rbf(X_train_scaled, y_train_onehot, centers, eta, max_epochs, s)
    weights_history.append(weights.copy())  # Save a copy of weights
    # Testar a RBF
    predictions = test_rbf(X_test_scaled, centers, weights, s)

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