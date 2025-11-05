# Importa NumPy para operações numéricas vetorizadas e álgebra linear.
import numpy as np
# Importa MLflow para rastrear parâmetros e métricas do experimento.
import mlflow

# Define (ou seleciona) o experimento onde os runs serão registrados.
mlflow.set_experiment("Técnicas de Otimização - SGD")

# Inicia um run no MLflow; tudo logado dentro deste bloco fica associado a esta execução.
with mlflow.start_run():
    # Função de ativação ReLU: zera valores negativos e mantém os positivos.
    def relu(x):
        return np.maximum(0, x)

    # Derivada da ReLU: 1 para x>0 e 0 caso contrário (no 0 usa-se 0 por convenção).
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Função de perda MSE: média dos quadrados das diferenças entre y verdadeiro e previsto.
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Matriz de entradas (duas amostras com duas features cada).
    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    # Saídas alvo para as duas amostras (coluna única).
    y_true = np.array([[0.3], [0.9]])

    # Pesos iniciais do modelo linear (2×1).
    weights = np.array([[0.1], [0.2]])

    # Taxa de aprendizado (passo do gradiente) e número de épocas.
    learning_rate = 0.2
    epochs = 100

    # Histórico para inspecionar a evolução dos pesos ao longo do treino.
    weight_history = []

    # Loop de treinamento por épocas usando SGD clássico (batch completo neste exemplo).
    for epoch in range(epochs):
        # Forward: pré-ativação z = X·W (shape 2×1).
        z = np.dot(x, weights)
        # Aplica ReLU para obter as predições (não linearidade).
        y_pred = relu(z)

        # Calcula o erro atual via MSE (escalar).
        error = mse_loss(y_true, y_pred)

        # Backward: gradiente em relação a W = Xᵀ · (ReLU'(z) ⊙ (y_pred − y_true)).
        gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))

        # Atualização dos pesos por descida do gradiente: W ← W − η·∇W.
        weights -= learning_rate * gradient
        # Armazena o snapshot dos pesos (referência; para cópia use weights.copy()).
        weight_history.append(weights)
        # Mostra os pesos atuais (útil para depuração rápida).
        print(weights)

        # Loga a métrica de erro no MLflow (nomes distintos por época; alternativa: usar step=epoch).
        mlflow.log_metric(f"Current error on Epoch {epoch}", error)
        # Loga o mesmo erro com outro nome (duplicado intencional do seu exemplo).
        mlflow.log_metric(f"error_epoch_{epoch}", error)
    
    # Exibe o histórico bruto de pesos coletado durante o treinamento.
    print(weight_history)
    # Mostra o valor final dos pesos aprendidos com SGD.
    print("Pesos finais com SGD:", weights)
    # Registra os pesos finais como parâmetro do run (serializado como lista).
    mlflow.log_param("final_weights_sgd", weights.tolist())
