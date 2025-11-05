# Tipos utilitários para anotar a assinatura (função de ativação e sua derivada).
from typing import Callable, Tuple

# Biblioteca numérica para operações vetorizadas e álgebra linear.
import numpy as np
# Plataforma de rastreamento de experimentos (parâmetros, métricas, artefatos).
import mlflow

# Importa ativação ReLU e sua derivada de um módulo local.
from relu import relu, relu_derivative
# Importa ativação Sigmóide e sua derivada de um módulo local.
from sigmoid import sigmoid, sigmoid_derivative
# Importa ativação Tanh e sua derivada de um módulo local.
from tanh import tanh, tanh_derivative


# Função de perda: erro quadrático médio (MSE) entre verdade e predição.
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Executa um passo simples de “backpropagation” para uma ativação/derivada fornecida.
def run_backpropagation(algorithm: Tuple[Callable, Callable]):
    # Seleciona/cria um experimento no MLflow com o nome da ativação.
    mlflow.set_experiment(f"Backpropagation for {algorithm[0].__name__}")
    # Abre um run no MLflow (tudo logado dentro ficará associado a esta execução).
    with mlflow.start_run():
        # Define dados de exemplo: entradas (2×2), rótulos verdadeiros (2×1) e pesos iniciais (2×1).
        x = np.array([[0.1, 0.2], [0.4, 0.5]])
        y_true = np.array([[0.3], [0.9]])
        weights = np.array([[0.1], [0.2]])

        # Passagem direta: pré-ativação z = X·W (resulta 2×1).
        z = np.dot(x, weights)
        # Aplica a função de ativação escolhida à pré-ativação (predição 2×1).
        y_pred = algorithm[0](z)

        # Calcula o erro inicial via MSE (escalar).
        error = mse_loss(y_true, y_pred)
        # Mostra o erro para inspeção.
        print("Erro inicial:", error)

        # Passagem reversa: gradiente de W = Xᵀ · (f'(z) ⊙ (y_pred − y_true)).
        gradient = np.dot(x.T, algorithm[1](z) * (y_pred - y_true))

        # Loga a métrica de erro inicial (float) no MLflow.
        mlflow.log_metric("initial_error", error)
        # Loga pesos e gradiente como parâmetros (serializados como lista).
        mlflow.log_param("initial_weights", weights.tolist())
        mlflow.log_param("gradient", gradient.tolist())

        # Atualização dos pesos por descida do gradiente com taxa de aprendizado fixa.
        learning_rate = 0.01
        weights -= learning_rate * gradient

        # Exibe os pesos atualizados após um passo.
        print("Pesos atualizados:", weights)
        # Registra os pesos atualizados no MLflow.
        mlflow.log_param("updated_weights", weights.tolist())
    # Retorna os pesos após a atualização.
    return weights


# Bloco principal: executa o experimento para Sigmóide, ReLU e Tanh.
if __name__ == "__main__":
    # Tuplas (ativação, derivada) a serem testadas.
    algorithms = [(sigmoid, sigmoid_derivative), (relu, relu_derivative), (tanh, tanh_derivative)]
    # Roda o procedimento para cada ativação configurada.
    for algo in algorithms:
        run_backpropagation(algo)
