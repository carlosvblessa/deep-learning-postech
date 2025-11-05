# Importa o NumPy para operações numéricas vetorizadas.
import numpy as np
# Importa o MLflow para rastrear parâmetros/métricas de experimentos.
import mlflow

# Define/seleciona o experimento no MLflow com este nome.
mlflow.set_experiment("Técnicas de Otimização - ADAM")

# Inicia um run do MLflow; tudo logado dentro deste bloco pertence a esta execução.
with mlflow.start_run():
    # Função de ativação ReLU (mantém positivos e zera negativos).
    def relu(x):
        return np.maximum(0, x)

    # Derivada da ReLU (1 para x>0, 0 caso contrário).
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    # Função de perda MSE (erro quadrático médio).
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Matriz de entradas (2 amostras × 2 atributos).
    x = np.array([[0.1, 0.2], [0.4, 0.5]])
    # Saídas-alvo (2 amostras × 1).
    y_true = np.array([[0.3], [0.9]])

    # Pesos iniciais do modelo (2 × 1).
    weights = np.array([[0.1], [0.2]])

    # Taxa de aprendizado do Adam.
    learning_rate = 0.2
    # Beta1: decaimento do momento de 1ª ordem.
    beta1 = 0.9
    # Beta2: decaimento do momento de 2ª ordem.
    beta2 = 0.999
    # Epsilon: termo de estabilidade numérica.
    epsilon = 1e-8
    # m: momento de 1ª ordem (inicialmente zeros, mesmo shape dos pesos).
    m = np.zeros_like(weights)
    # v: momento de 2ª ordem (inicialmente zeros, mesmo shape dos pesos).
    v = np.zeros_like(weights)
    # Passo temporal (para correção de viés).
    t = 0
    # Número de épocas de treinamento.
    epochs = 100

    # Laço de treinamento por épocas.
    for epoch in range(epochs):
        # Avança o passo temporal do Adam.
        t += 1
        # Forward linear: pré-ativação z = X·W.
        z = np.dot(x, weights)
        # Saída após ativação ReLU.
        y_pred = relu(z)
        # Erro atual via MSE (escalar).
        error = mse_loss(y_true, y_pred)
        # Gradiente do erro em relação aos pesos (considerando ReLU').
        gradient = np.dot(x.T, relu_derivative(z) * (y_pred - y_true))

        # Atualiza o momento de 1ª ordem (m).
        m = beta1 * m + (1 - beta1) * gradient
        # Atualiza o momento de 2ª ordem (v).
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Correção de viés para m (m chapéu).
        m_hat = m / (1 - beta1 ** t)
        # Correção de viés para v (v chapéu).
        v_hat = v / (1 - beta2 ** t)

        # Atualização dos pesos segundo a regra do Adam.
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Loga a métrica de erro desta época no MLflow.
        mlflow.log_metric(f"error_epoch_{epoch}", error)

    # Exibe os pesos finais aprendidos com Adam.
    print("Pesos finais com ADAM:", weights)
    # Registra os pesos finais como parâmetro do experimento.
    mlflow.log_param("final_weights_adam", weights.tolist())
