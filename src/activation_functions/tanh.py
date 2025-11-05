# Importa o NumPy para operações numéricas vetorizadas.
import numpy as np
# Importa o MLflow para rastrear parâmetros e métricas do experimento.
import mlflow

# Define/seleciona o experimento no MLflow (cria se não existir).
mlflow.set_experiment("Funções de Ativação - Tanh")

# Implementa a função de ativação tanh: mapeia valores para o intervalo (-1, 1).
def tanh(vector):
    return np.tanh(vector)

# Implementa a derivada da tanh: 1 − tanh(x)^2 (elemento a elemento).
def tanh_derivative(vector):
    return 1 - np.tanh(vector)**2

# Inicia uma execução (run) no MLflow para logar parâmetros e métricas deste bloco.
with mlflow.start_run():

    # Define um exemplo de entrada (array 2D) com valores inteiros.
    x = np.array(
        [
            [0, 255, 255], [0, 150, 255], [100, 0, 150],
            [0, 255, 255], [0, 150, 255], [100, 0, 150],
            [0, 255, 255], [0, 150, 255], [100, 0, 150]
        ]
    )
    # Aplica a função tanh ao array x (saída no intervalo (-1, 1)).
    tanh_output = tanh(x)
    # Calcula a derivada da tanh para os mesmos valores de x.
    tanh_grad = tanh_derivative(x)

    # Registra o parâmetro de entrada (convertido para listas) no MLflow.
    mlflow.log_param("input", x.tolist())
    # Loga a média da saída de tanh como métrica.
    mlflow.log_metric("tanh_output", tanh_output.mean())
    # Loga a média da derivada da tanh como métrica.
    mlflow.log_metric("tanh_derivative", tanh_grad.mean())

    # Imprime no console a saída da tanh para inspeção.
    print("Saída da Tanh:", tanh_output)
    # Imprime no console a derivada correspondente da tanh.
    print("Derivada da Tanh:", tanh_grad)
