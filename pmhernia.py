import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath, delimiter=r'\s+', header=None)
    data = data.dropna()  # Limpeza de dados
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Codificar rótulos categóricos
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded

# Função de ativação para o Perceptron Simples
def step_function(x):
    return np.where(x >= 0, 1, 0)


# Inicializar os pesos com valores aleatórios
def initialize_weights(input_dim):
    return np.random.uniform(-0.5, 0.5, input_dim + 1)


# Função para o Forward pass
def forward_pass(X, weights):
    return step_function(np.dot(X, weights.T))


# Função para treinamento do Perceptron Simples
def train_perceptron(X, y, weights, eta, max_epochs):
    start_time = time.time()
    for epoch in range(max_epochs):
        errors = 0
        for i in range(X.shape[0]):
            y_pred = forward_pass(X[i], weights)
            error = y[i] - y_pred
            weights += eta * error * X[i]
            errors += int(error != 0)
        if errors == 0:
            break
    training_time = time.time() - start_time
    return weights, training_time


# Função para teste do Perceptron Simples
def test_perceptron(X, weights):
    y_pred = forward_pass(X, weights)
    return y_pred



def process_dataset(filepath):
    X, y = load_and_prepare_data(filepath)

    # Parâmetros do Perceptron Simples
    eta = 0.01  # Taxa de aprendizagem
    max_epochs = 1000  # Número máximo de épocas

    # Realiza as 100 rodadas de treinamento e teste
    accuracies = []
    specificities = []
    sensitivities = []
    weights_history = []  # List to store weights for each round
    training_times = []
    confusion_matrices = []
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

        # Inicializa os pesos do Perceptron Simples
        weights = initialize_weights(X_train_scaled.shape[1] - 1)

        # Treinar o Perceptron Simples
        trained_weights,train_time  = train_perceptron(X_train_scaled, y_train, weights, eta, max_epochs)
        weights_history.append(trained_weights.copy())  # Save a copy of weights
        # Testar o Perceptron Simples
        predictions = test_perceptron(X_test_scaled, trained_weights)

        # Matrizes de Confusão
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        confusion = confusion_matrix(y_test, predictions)
        confusion_matrices.append(confusion)

        # Calcula medidas
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

        # Armazena resultados
        accuracies.append(accuracy)
        specificities.append(specificity)
        sensitivities.append(sensitivity)
        training_times.append(train_time)

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

    # Analisar e exibir matrizes de confusão
    min_accuracy_index = np.argmin(accuracies)
    max_accuracy_index = np.argmax(accuracies)
    print("Confusion Matrix with Lowest Accuracy:\n", confusion_matrices[min_accuracy_index])
    print("Confusion Matrix with Highest Accuracy:\n", confusion_matrices[max_accuracy_index])

    # Plotting box-plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].boxplot(accuracies)
    axs[0].set_title('Accuracy')
    axs[1].boxplot(specificities)
    axs[1].set_title('Specificity')
    axs[2].boxplot(sensitivities)
    axs[2].set_title('Sensitivity')
    plt.show()

    print(np.mean(accuracies), np.mean(training_times))

# Caminhos dos arquivos
filepath_2c = r'C:\Users\pgsmc\PycharmProjects\pythonProject5\column_2C.dat'
filepath_3c = r'C:\Users\pgsmc\PycharmProjects\pythonProject5\column_3C.dat'

# Processamento de ambos os conjuntos de dados
accuracy_2c = process_dataset(filepath_2c)
accuracy_3c = process_dataset(filepath_3c)
