# Importa o NumPy para operações numéricas vetorizadas.
import numpy as np
# Importa o MLflow para rastrear parâmetros/métricas do experimento.
import mlflow


# Define (ou cria) o experimento no MLflow com este nome.
mlflow.set_experiment("Funções de Ativação - Sigmóide")

# Implementa a função sigmóide: σ(x) = 1 / (1 + e^{-x}), aplicada elemento a elemento.
def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

# Derivada da sigmóide: σ'(x) = σ(x) * (1 − σ(x)), também elemento a elemento.
def sigmoid_derivative(vector):
    sig = sigmoid(vector)
    return sig * (1 - sig)

# Inicia um "run" no MLflow; tudo que for logado ficará associado a esta execução.
with mlflow.start_run():

    # Exemplo de entrada: matriz com valores negativos, zero e positivos.
    x = np.array(
        [
            [-1, 0, 1], [-2, 5, 2], [100, 0, -150],
            [-1, 0, 1], [-2, 5, 2], [500, 0, -100],
            [-1, 0, 1], [-2, 5, 2], [125, 0, -214]
        ]
    )
    # Aplica a sigmóide ao array x (resultado entre 0 e 1).
    sigmoid_output = sigmoid(x)
    # Calcula a derivada da sigmóide para os mesmos valores.
    sigmoid_grad = sigmoid_derivative(x)

    # Registra a entrada original como parâmetro (convertida para listas).
    mlflow.log_param("input", x.tolist())
    # Loga a média da saída da sigmóide como métrica.
    mlflow.log_metric("sigmoid_output", sigmoid_output.mean())
    # Loga a média da derivada da sigmóide como métrica.
    mlflow.log_metric("sigmoid_derivative", sigmoid_grad.mean())

    # Exibe no console a saída da sigmóide para inspeção.
    print("Saída da Sigmóide:", sigmoid_output)
    # Exibe no console a derivada correspondente.
    print("Derivada da Sigmóide:", sigmoid_grad)
