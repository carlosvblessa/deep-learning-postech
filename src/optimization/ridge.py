# Importa NumPy para operações numéricas vetorizadas.
import numpy as np
# Importa MLflow para rastrear experimentos (métricas/parâmetros).
import mlflow

# Define/seleciona o experimento no MLflow com este nome.
mlflow.set_experiment("Técnicas de Regularização RIDGE")

# Inicia um run no MLflow; tudo logado dentro será associado a esta execução.
with mlflow.start_run():
    # Função de ativação ReLU: max(0, x) elemento a elemento.
    def relu(x):
        return np.maximum(0, x)

    # Derivada da ReLU: 1 se x>0, caso contrário 0 (no 0, por convenção, 0).
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Função de perda MSE: média do erro quadrático.
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Penalidade L2 (Ridge): λ * ||W||² (soma dos quadrados dos pesos).
    def l2_regularization(weights, lambd):
        return lambd * np.sum(weights ** 2)

    # Matriz de entradas (2 amostras × 2 features).
    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    # Saídas-alvo correspondentes (2 amostras × 1).
    y_true = np.array([[0.3], [0.9]])

    # Pesos iniciais do modelo (2 × 1).
    weights = np.array([[0.1], [0.2]])
    # Taxa de aprendizado para atualização por gradiente.
    learning_rate = 0.01
    # Coeficiente de regularização L2 (λ).
    lambd = 0.01  # Coeficiente de regularização L2
    # Número de épocas de treinamento.
    epochs = 100

    # Loop de treinamento por épocas.
    for epoch in range(epochs):
        # Forward linear: z = X·W.
        z = np.dot(x, weights)
        # Aplica ReLU para obter a predição.
        y_pred = relu(z)
        # Calcula o erro de ajuste (MSE).
        error = mse_loss(y_true, y_pred)
        # Calcula a penalidade L2 (apenas o valor escalar).
        regularization_loss = l2_regularization(weights, lambd)
        # Perda total = erro de ajuste + penalidade de regularização.
        total_loss = error + regularization_loss

        # Gradiente do termo de ajuste (sem incluir o termo L2).
        # OBS: para Ridge “completo”, somar também 2*λ*W ao gradiente.
        gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))
        # Atualiza os pesos por descida do gradiente.
        weights -= learning_rate * gradient

        # Registra a perda total desta época no MLflow (como métrica).
        mlflow.log_metric(f"total_loss_epoch_{epoch}", total_loss)

    # Exibe os pesos finais após o treinamento.
    print("Pesos finais com L2:", weights)
    # Registra os pesos finais como parâmetro do run.
    mlflow.log_param("final_weights_l2", weights.tolist())
