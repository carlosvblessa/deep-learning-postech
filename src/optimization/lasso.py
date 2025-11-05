# Importa NumPy para operações numéricas vetorizadas.
import numpy as np
# Importa MLflow para rastrear parâmetros/métricas dos experimentos.
import mlflow

# Define (ou seleciona) o experimento no MLflow com este nome.
mlflow.set_experiment("Técnicas de Regularização LASSO")

# Inicia um run no MLflow; tudo logado dentro ficará associado a esta execução.
with mlflow.start_run():
    # Função de ativação ReLU: aplica max(0, x) elemento a elemento.
    def relu(x):
        return np.maximum(0, x)

    # Derivada da ReLU: 1 onde x>0, 0 caso contrário (no 0 costuma-se adotar 0).
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Função de perda MSE: média dos erros quadráticos.
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Regularização L1 (LASSO): λ * soma dos módulos dos pesos.
    def l1_regularization(weights, lambd):
        return lambd * np.sum(np.abs(weights))

    # Entradas de exemplo (2 amostras × 2 features).
    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    # Saídas-alvo correspondentes (2 amostras × 1).
    y_true = np.array([[0.3], [0.9]])

    # Pesos iniciais (2 × 1) para o modelo linear.
    weights = np.array([[0.1], [0.2]])
    # Taxa de aprendizado para a atualização dos pesos.
    learning_rate = 0.01
    # Coeficiente de regularização L1 (λ).
    lambd = 0.01  # Coeficiente de regularização L1
    # Número de épocas de treinamento.
    epochs = 100

    # Loop de treinamento por épocas.
    for epoch in range(epochs):
        # Passagem direta: pré-ativação z = X·W.
        z = np.dot(x, weights)
        # Aplica ReLU para obter a predição.
        y_pred = relu(z)
        # Calcula erro MSE entre predição e alvo.
        error = mse_loss(y_true, y_pred)
        # Calcula termo de regularização L1 (apenas o valor).
        regularization_loss = l1_regularization(weights, lambd)
        # Perda total = erro de ajuste + penalidade L1.
        total_loss = error + regularization_loss

        # Gradiente do erro de ajuste (sem o subgradiente L1):
        # OBS: aqui a penalidade L1 NÃO está sendo incorporada ao gradiente (faltaria λ*sign(W)).
        gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))
        # Atualiza os pesos por descida do gradiente.
        weights -= learning_rate * gradient

        # Registra a perda total no MLflow; OBS: cada época vira uma métrica com nome distinto.
        # (Alternativa comum seria usar `mlflow.log_metric("total_loss", total_loss, step=epoch)`.)
        mlflow.log_metric(f"total_loss_epoch_{epoch}", total_loss)

    # Exibe os pesos finais após o treinamento.
    print("Pesos finais com L1:", weights)
    # Loga os pesos finais como parâmetro no MLflow (serializados em lista).
    mlflow.log_param("final_weights_l1", weights.tolist())
