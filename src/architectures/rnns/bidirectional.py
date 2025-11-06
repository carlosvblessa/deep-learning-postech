# Importa o PyTorch principal para tensores e operações.
import torch
# Importa módulos de redes neurais (camadas, modelos).
import torch.nn as nn
# Importa otimizadores (Adam, SGD, etc.).
import torch.optim as optim
# Importa utilitários para criar DataLoaders e datasets baseados em tensores.
from torch.utils.data import DataLoader, TensorDataset
# Importa NumPy para operações numéricas auxiliares.
import numpy as np
# Importa MLflow para rastrear métricas e parâmetros de experimento.
import mlflow
# Extensão do MLflow para salvar/carregar modelos PyTorch.
import mlflow.pytorch

# Define o dispositivo de execução: usa CUDA se disponível, caso contrário CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensionalidade de entrada (número de features por passo temporal).
input_size = 10      # Number of features in the input data
# Tamanho do estado oculto da RNN (número de unidades por direção).
hidden_size = 50     # Number of hidden units in the RNN
# Número de camadas empilhadas na RNN.
num_layers = 2       # Number of RNN layers
# Dimensionalidade da saída (regressão escalar neste exemplo).
output_size = 1      # Number of output units (e.g., regression output)
# Número de épocas de treinamento.
num_epochs = 50
# Tamanho do minibatch.
batch_size = 64
# Taxa de aprendizado do otimizador.
learning_rate = 0.001
# Comprimento das sequências de entrada (número de passos de tempo).
sequence_length = 20  # Length of the input sequences
# Número de amostras artificiais a gerar para treino.
num_samples = 10000  # Number of artificial samples to generate

# Gera dados artificiais: sequências aleatórias X e alvos y (regressão).
def generate_artificial_data(num_samples, sequence_length, input_size):
    # Cria tensores aleatórios com formato (N, T, F).
    X = torch.randn(num_samples, sequence_length, input_size)
    
    # Cria alvos aleatórios contínuos (regressão) com formato (N, 1).
    y = torch.randn(num_samples, 1)
    
    # Retorna pares (X, y) para uso em datasets.
    return X, y

# Gera conjunto de treino (N amostras).
train_X, train_y = generate_artificial_data(num_samples, sequence_length, input_size)
# Gera conjunto de teste (N/10 amostras).
test_X, test_y = generate_artificial_data(num_samples // 10, sequence_length, input_size)

# Empacota tensores em um TensorDataset para treino.
train_dataset = TensorDataset(train_X, train_y)
# Empacota tensores em um TensorDataset para teste.
test_dataset = TensorDataset(test_X, test_y)
# Cria DataLoader de treino com embaralhamento.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# Cria DataLoader de teste sem embaralhar.
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define uma RNN bidirecional simples para regressão em sequência.
class BiRNN(nn.Module):
    # Construtor com hiperparâmetros principais.
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        # Inicializa a superclasse nn.Module.
        super().__init__()
        # Armazena tamanho do estado oculto para criar h0.
        self.hidden_size = hidden_size
        # Armazena número de camadas (por direção).
        self.num_layers = num_layers
        # Cria RNN clássica (tanh), bidirecional, com batch_first=(N,T,F).
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Camada linear final; *2 por causa das duas direções (fwd+rev).
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    # Define o fluxo direto da rede.
    def forward(self, x):
        # Estado oculto inicial: (num_layers*2, batch, hidden_size).
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        # Executa a RNN ao longo do tempo; out tem forma (N, T, 2*hidden).
        out, _ = self.rnn(x, h0)
        
        # Usa apenas o último passo temporal e projeta para a saída.
        out = self.fc(out[:, -1, :])
        # Retorna previsão (regressão escalar por amostra).
        return out

# Treina o modelo e registra métricas/artefatos no MLflow.
def train_model():
    # Instancia a BiRNN e move para o device apropriado.
    model = BiRNN(input_size, hidden_size, num_layers, output_size).to(device)
    # Perda MSE adequada para tarefa de regressão.
    criterion = nn.MSELoss()
    # Otimizador Adam com taxa de aprendizado definida.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define/seleciona o experimento no MLflow.
    mlflow.set_experiment("BiRNN Artificial Data Regression")
    # Inicia um run para logar parâmetros, métricas e modelo.
    with mlflow.start_run():
        # Loga hiperparâmetros principais do experimento.
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Loop de treinamento por épocas.
        for epoch in range(num_epochs):
            # Coloca a rede em modo treinamento.
            model.train()
            # Acumulador de perda média da época.
            running_loss = 0.0
            
            # Itera sobre os minibatches do DataLoader.
            for i, (sequences, labels) in enumerate(train_loader):
                # Move entradas e rótulos para o device.
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward pass: obtém a saída prevista.
                outputs = model(sequences)
                # Calcula a perda do lote.
                loss = criterion(outputs, labels)

                # Zera gradientes, faz backprop e atualiza os pesos.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Acumula a perda para relatório da época.
                running_loss += loss.item()
                
                # Loga a cada 100 lotes: progresso e loss média corrente.
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)

        # Salva o modelo treinado como artefato do MLflow.
        mlflow.pytorch.log_model(model, "birnn_artificial_data_model")

        # Avalia o modelo no conjunto de teste e loga a métrica.
        evaluate_model(model, criterion)

# Avalia o modelo em teste calculando a perda média (MSE).
def evaluate_model(model, criterion):
    # Modo de avaliação (desativa dropout, etc.).
    model.eval()
    # Acumulador da soma das perdas dos lotes.
    test_loss = 0.0
    # Sem gradientes para acelerar e poupar memória.
    with torch.no_grad():
        # Percorre o DataLoader de teste.
        for sequences, labels in test_loader:
            # Move dados para o device.
            sequences, labels = sequences.to(device), labels.to(device)
            # Forward: obtém previsões.
            outputs = model(sequences)
            # Calcula a perda do lote e acumula.
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Calcula a perda média final no conjunto de teste.
    average_test_loss = test_loss / len(test_loader)
    # Exibe a métrica de avaliação.
    print(f"Test Loss: {average_test_loss:.4f}")
    # Loga a métrica no MLflow.
    mlflow.log_metric("test_loss", average_test_loss)

# Ponto de entrada: executa o treinamento e a avaliação.
train_model()
