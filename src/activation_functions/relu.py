# Importa o NumPy para operações vetorizadas e manipulação de arrays.
import numpy as np
# Importa o MLflow para rastreamento de experimentos (parâmetros, métricas, artefatos).
import mlflow


# Define/seleciona o experimento no MLflow (cria se não existir).
mlflow.set_experiment("Funções de Ativação - ReLU")

# Implementa a função de ativação ReLU: max(0, x) elemento a elemento.
def relu(vector):
    return np.maximum(0, vector)

# Implementa a derivada da ReLU: 1 onde x>0 e 0 caso contrário (no ponto 0 costuma-se usar 0).
def relu_derivative(vector):
    return np.where(vector > 0, 1, 0)


# Inicia um run no MLflow; tudo que for logado dentro deste bloco ficará associado a este run.
with mlflow.start_run():
    # Exemplo de entrada: matriz 3×3 repetida 3 vezes (total 9×3) com valores negativos, zero e positivos.
    x = np.array(
        [
            [-1, 0, 1], [-2, 5, 2], [100, 0, -100],
            [-1, 0, 1], [-2, 5, 2], [100, 0, -100],
            [-1, 0, 1], [-2, 5, 2], [100, 0, -100]
        ]
    )
    # Aplica a ReLU de forma vetorizada; negativos viram 0, positivos permanecem.
    relu_output = relu(x)
    # Calcula a “derivada” da ReLU para cada posição (1 se valor>0, senão 0).
    relu_grad = relu_derivative(x)

    # Loga o parâmetro de entrada (como lista) para registro/reprodutibilidade.
    mlflow.log_param("input", x.tolist())
    # Loga a métrica: média dos valores após ReLU (scalar).
    mlflow.log_metric("relu_output", relu_output.mean())
    # Loga a métrica: média da derivada (proporção de elementos positivos).
    mlflow.log_metric("relu_derivative", relu_grad.mean())

    # Imprime no console a saída da ReLU para inspeção.
    print("Saída da ReLU:", relu_output)
    # Imprime no console a derivada da ReLU correspondente.
    print("Derivada da ReLU:", relu_grad)
