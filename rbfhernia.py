
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

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada da função sigmoid
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
    start_time = time.time()
    activations = calculate_rbf_activations(X, centers, s)
    weights = initialize_weights(activations.shape[1], y.shape[1])
    for epoch in range(max_epochs):
        for i in range(X.shape[0]):
            y_pred = forward_pass(activations[i].reshape(1, -1), weights)
            error = y[i] - y_pred
            weights += eta * error.T.dot(np.insert(activations[i], 0, 1).reshape(1, -1))
    training_time = time.time() - start_time
    return weights, training_time


# Função para teste da RBF
def test_rbf(X, centers, weights, s):
    activations = calculate_rbf_activations(X, centers, s)
    y_pred = forward_pass(activations, weights)
    return np.argmax(y_pred, axis=1)




def process_dataset(filepath,output_dim ):
    X, y = load_and_prepare_data(filepath)



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
    training_times = []
    confusion_matrices = []

    for _ in range(10):
        # Divisão aleatória dos dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)
        y_train_onehot = np.zeros((y_train.size, 3))
        y_train_onehot[np.arange(y_train.size), y_train] = 1

        # Normalização dos dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Ajustar os centros
        centers = adjust_centers(X_train_scaled, q, eta, max_iterations)

        # Treinar a RBF
        weights,train_time  = train_rbf(X_train_scaled, y_train_onehot, centers, eta, max_epochs, s)
        weights_history.append(weights.copy())  # Save a copy of weights
        # Testar a RBF
        predictions = test_rbf(X_test_scaled, centers, weights, s)

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
#accuracy_2c = process_dataset(filepath_2c,2)
accuracy_3c = process_dataset(filepath_3c,3)