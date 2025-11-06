# Importa PyTorch base para tensores e operações numéricas de baixo nível.
import torch
# Importa os módulos de redes neurais (camadas, losses, etc.).
import torch.nn as nn
# Importa otimizadores (Adam, SGD, etc.).
import torch.optim as optim
# Importa utilitários para DataLoader e dataset baseado em tensores.
from torch.utils.data import DataLoader, TensorDataset
# Importa NumPy para apoio em operações numéricas.
import numpy as np
# Importa o MLflow para rastrear experimentos (métricas, parâmetros).
import mlflow
# Extensão do MLflow para salvar/carregar modelos PyTorch.
import mlflow.pytorch

# Define o dispositivo de execução: GPU (CUDA) se disponível, senão CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensão das features por passo temporal (entrada da GRU).
input_size = 10      # Number of features in the input data
# Tamanho do estado oculto da GRU (número de unidades).
hidden_size = 50     # Number of hidden units in the GRU
# Número de camadas empilhadas na GRU.
num_layers = 2       # Number of GRU layers
# Dimensão da saída (regressão escalar neste exemplo).
output_size = 1      # Number of output units (e.g., regression output)
# Quantidade de épocas de treinamento.
num_epochs = 50
# Tamanho do minibatch.
batch_size = 64
# Taxa de aprendizado do otimizador.
learning_rate = 0.001
# Comprimento das sequências (passos de tempo).
sequence_length = 20  # Length of the input sequences
# Quantidade de amostras artificiais a serem geradas para treino.
num_samples = 10000  # Number of artificial samples to generate

# Função geradora de dados artificiais de sequência (X) e alvos (y).
def generate_artificial_data(num_samples, sequence_length, input_size):
    # Cria tensor de entradas com shape (N, T, F), amostrado de N(0,1).
    X = torch.randn(num_samples, sequence_length, input_size)
    
    # Cria alvos contínuos para regressão com shape (N, 1).
    y = torch.randn(num_samples, 1)
    
    # Retorna o par (entradas, alvos).
    return X, y

# Gera o conjunto de treino completo.
train_X, train_y = generate_artificial_data(num_samples, sequence_length, input_size)
# Gera o conjunto de teste com 1/10 do tamanho.
test_X, test_y = generate_artificial_data(num_samples // 10, sequence_length, input_size)

# Empacota tensores de treino em um TensorDataset.
train_dataset = TensorDataset(train_X, train_y)
# Empacota tensores de teste em um TensorDataset.
test_dataset = TensorDataset(test_X, test_y)
# Cria DataLoader de treino com embaralhamento.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# Cria DataLoader de teste sem embaralhar.
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define o modelo baseado em GRU para regressão.
class GRU(nn.Module):
    # Construtor que recebe tamanhos de entrada/oculto/saída e nº de camadas.
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        # Inicializa a superclasse nn.Module.
        super().__init__()
        # Guarda o tamanho do estado oculto para montar h0.
        self.hidden_size = hidden_size
        # Guarda o número de camadas (empilhadas).
        self.num_layers = num_layers

        # Camada recorrente GRU (batch_first=True → shape (N, T, F)).
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Projeção linear do último hidden state para a saída.
        self.fc = nn.Linear(hidden_size, output_size)
        # Define uma sigmoide (não usada neste fluxo de regressão).
        self.sigmoid = nn.Sigmoid()

        # Segunda GRU definida mas não utilizada (código redundante).
        self.gru2 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Segunda projeção linear também não utilizada.
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Redefinição da sigmoide (sobrescreve a anterior).
        self.sigmoid = nn.Sigmoid()

    # Passo forward: recebe lote de sequências e produz saída por amostra.
    def forward(self, x):
        # Estado inicial h0 com zeros: shape (num_layers, N, hidden_size).
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Propagação na GRU principal ao longo do tempo (ignora gru2).
        out, _ = self.gru(x, h0)
        # Seleciona a saída do último passo temporal e projeta na fc.
        out = self.fc(out[:, -1, :])
        # Retorna a predição (regressão escalar).
        return out

# Função de treinamento do modelo com logging no MLflow.
def train_model():
    # Instancia o modelo e move para o dispositivo escolhido.
    model = GRU(input_size, hidden_size, num_layers, output_size).to(device)
    # Função de perda MSE apropriada para regressão.
    criterion = nn.MSELoss()
    # Otimizador Adam com a taxa de aprendizado definida.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Seleciona/cria o experimento no MLflow.
    mlflow.set_experiment("GRU Artificial Data Regression")
    # Inicia uma execução (run) para logar tudo deste treino.
    with mlflow.start_run():
        # Loga hiperparâmetros relevantes do experimento.
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Loop principal de épocas de treino.
        for epoch in range(num_epochs):
            # Coloca o modelo em modo treino.
            model.train()
            # Acumulador da perda média por época.
            running_loss = 0.0
            
            # Itera sobre os minibatches do DataLoader.
            for i, (sequences, labels) in enumerate(train_loader):
                # Move dados para o device (GPU/CPU).
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward pass: calcula as previsões.
                outputs = model(sequences)
                # Calcula a perda do lote atual.
                loss = criterion(outputs, labels)

                # Zera gradientes, retropropaga e atualiza os pesos.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Atualiza a perda acumulada da época.
                running_loss += loss.item()
                
                # Faz logging periódico de progresso e métrica no MLflow.
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)

        # Salva o modelo treinado como artefato do MLflow.
        mlflow.pytorch.log_model(model, "gru_artificial_data_model")

        # Avalia o modelo no conjunto de teste e loga a métrica.
        evaluate_model(model, criterion)

# Função de avaliação em teste (perda média MSE).
def evaluate_model(model, criterion):
    # Coloca o modelo em modo avaliação (desativa dropout, etc.).
    model.eval()
    # Acumulador da soma das perdas nos lotes de teste.
    test_loss = 0.0
    # Desativa gradientes para acelerar e economizar memória.
    with torch.no_grad():
        # Itera sobre os lotes do conjunto de teste.
        for sequences, labels in test_loader:
            # Move dados para o device.
            sequences, labels = sequences.to(device), labels.to(device)
            # Forward: obtém as previsões.
            outputs = model(sequences)
            # Calcula a perda do lote e acumula.
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Calcula a perda média ao final.
    average_test_loss = test_loss / len(test_loader)
    # Exibe a métrica final de teste.
    print(f"Test Loss: {average_test_loss:.4f}")
    # Loga a perda de teste no MLflow.
    mlflow.log_metric("test_loss", average_test_loss)

# Executa o fluxo completo: treino + avaliação.
train_model()
