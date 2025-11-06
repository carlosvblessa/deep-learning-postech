# Importa o PyTorch base para tensores e operações numéricas.
import torch
# Importa módulos de redes neurais (camadas, perdas, etc.).
import torch.nn as nn
# Importa otimizadores (Adam, SGD, etc.).
import torch.optim as optim
# DataLoader e TensorDataset para empacotar dados em lotes.
from torch.utils.data import DataLoader, TensorDataset
# NumPy para utilidades numéricas.
import numpy as np
# MLflow para rastrear experimentos (parâmetros/métricas).
import mlflow
# Integração do MLflow específica para modelos PyTorch.
import mlflow.pytorch

# Seleciona o device: GPU (CUDA) se disponível; caso contrário, CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensão das features por passo temporal na sequência de entrada.
input_size = 10      # Number of features in the input data
# Tamanho do estado oculto do LSTM (número de unidades).
hidden_size = 50     # Number of hidden units in the LSTM
# Número de camadas empilhadas do LSTM.
num_layers = 2       # Number of LSTM layers
# Dimensão da saída (aqui 1 — típico de regressão escalar).
output_size = 1      # Number of output units (e.g., regression output)
# Número de épocas de treinamento.
num_epochs = 50
# Tamanho do minibatch para treinamento e teste.
batch_size = 64
# Taxa de aprendizado do otimizador.
learning_rate = 0.001
# Comprimento (número de passos de tempo) das sequências.
sequence_length = 20  # Length of the input sequences
# Quantidade de amostras artificiais a gerar para o conjunto de treino.
num_samples = 10000  # Number of artificial samples to generate


# Função utilitária para gerar dados artificiais (sequências e alvos).
def generate_artificial_data(num_samples, sequence_length, input_size):
    # Cria tensor de entradas X ~ N(0,1) com shape (N, T, F).
    X = torch.randn(num_samples, sequence_length, input_size)

    # Cria alvos de regressão y ~ N(0,1) com shape (N, 1).
    y = torch.randn(num_samples, 1)

    # Retorna o par (X, y) para uso em TensorDataset.
    return X, y

# Gera o conjunto de treino completo com N amostras.
train_X, train_y = generate_artificial_data(num_samples, sequence_length, input_size)
# Gera o conjunto de teste com N/10 amostras.
test_X, test_y = generate_artificial_data(num_samples // 10, sequence_length, input_size)

# Empacota (X, y) de treino em um Dataset baseado em tensor.
train_dataset = TensorDataset(train_X, train_y)
# Empacota (X, y) de teste em um Dataset baseado em tensor.
test_dataset = TensorDataset(test_X, test_y)
# DataLoader de treino com embaralhamento (shuffle=True).
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# DataLoader de teste sem embaralhar (shuffle=False).
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Função que constrói um dicionário com camadas internas (para logging).
def get_inner_layrs(input_size, hidden_size, num_layers, output_size):
    # Retorna um dicionário com instâncias de camadas, indexadas por um nome string.
    return {
        str(nn.LSTM) + "_1": nn.LSTM(input_size, hidden_size, num_layers, batch_first=True),
        str(nn.Sigmoid) + "_1": nn.Sigmoid(),
        str(nn.Linear) + "_1": nn.Linear(hidden_size, output_size),
        str(nn.LSTM) + "_2": nn.LSTM(input_size, hidden_size, num_layers, batch_first=True),
        str(nn.Softmax) + "_1": nn.Softmax(dim=1),
        str(nn.Linear) + "_2": nn.Linear(hidden_size, output_size),
        str(nn.Softmax) + "_2": nn.Softmax(dim=1)
    }

# Define um modelo LSTM para processar sequências e projetar para uma saída.
class LSTM(nn.Module):
    # Construtor recebe tamanhos de entrada/oculto/saída e número de camadas.
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        # Inicializa a superclasse nn.Module.
        super().__init__()
        # Armazena o tamanho do estado oculto para inicializar h0/c0.
        self.hidden_size = hidden_size
        # Armazena o número de camadas empilhadas do LSTM.
        self.num_layers = num_layers

        # Camada LSTM principal (batch_first=True: entrada (N, T, F)).
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Cabeça sequencial de pós-processamento da saída do LSTM.
        self.model = nn.Sequential(
            nn.Sigmoid(),                 # Ativação não linear.
            nn.Linear(hidden_size, output_size),  # Projeção para 'output_size'.
            nn.Softmax(dim=1),            # Softmax ao longo de dim=1 (provável para classificação).
            nn.Linear(output_size, input_size),   # Projeção de volta para 'input_size'.
            nn.Softmax(dim=-1)            # Softmax na última dimensão (novamente, típico de classific.)
        )

    # Caminho direto (forward): recebe batch de sequências e devolve predições.
    def forward(self, x):
        # Obtém o tamanho do batch para moldar estados iniciais.
        batch_size = x.size(0)

        # Inicializa estados oculto (h0) e de célula (c0) com zeros.
        h0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0_1 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Executa o LSTM: 'out' tem shape (N, T, hidden_size).
        out, _ = self.lstm(x, (h0_1, c0_1))
        # Toma o último passo temporal e passa pelo cabeçalho sequencial.
        out = self.model(out[:, -1, :])

        # Retorna a predição final.
        return out


# Função de treinamento que organiza o loop, logging e avaliação.
def train_model():
    # Instancia o modelo e move para o dispositivo selecionado.
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    # Define a perda (MSE) apropriada para regressão contínua.
    criterion = nn.MSELoss()
    # Define o otimizador Adam com a LR indicada.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define (ou cria) o experimento no MLflow.
    mlflow.set_experiment("LSTM Artificial Data Regression")
    # Inicia uma execução de experimento (run) no MLflow.
    with mlflow.start_run():
        # Loga os nomes das camadas internas (chaves do dicionário).
        mlflow.log_param("intermediate_layers", [*get_inner_layrs(input_size, hidden_size, num_layers, output_size).keys()])
        # Loga hiperparâmetros relevantes do treinamento.
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Loop de épocas de treinamento.
        for epoch in range(num_epochs):
            # Coloca o modelo em modo de treino (ativa dropout/bn de treino).
            model.train()
            # Acumulador de perda média na época.
            running_loss = 0.0

            # Itera sobre minibatches do DataLoader de treino.
            for i, (sequences, labels) in enumerate(train_loader):
                # Move dados para o device (GPU/CPU).
                sequences, labels = sequences.to(device), labels.to(device)

                # Forward: gera predições para o lote atual.
                outputs = model(sequences)
                # Calcula a perda MSE entre predição e rótulos.
                loss = criterion(outputs, labels)

                # Zera gradientes, faz backprop e atualiza pesos.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Acumula a perda para estatística da época.
                running_loss += loss.item()
                
                # Faz logging periódico de métrica e imprime progresso.
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)

        # Salva o modelo treinado como artefato no MLflow.
        mlflow.pytorch.log_model(model, "lstm_artificial_data_model")

        # Avalia o modelo no conjunto de teste e loga a métrica.
        evaluate_model(model, criterion)

# Função de avaliação: calcula a perda média no conjunto de teste.
def evaluate_model(model, criterion):
    # Modo avaliação (desativa dropout e normalizações de treino).
    model.eval()
    # Acumulador de perda total em teste.
    test_loss = 0.0
    # Desliga autograd para acelerar inferência.
    with torch.no_grad():
        # Percorre os lotes do DataLoader de teste.
        for sequences, labels in test_loader:
            # Move dados para o device.
            sequences, labels = sequences.to(device), labels.to(device)
            # Forward: obtém predições do modelo.
            outputs = model(sequences)
            # Calcula perda do lote e acumula.
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Média da perda em todos os lotes de teste.
    average_test_loss = test_loss / len(test_loader)
    # Exibe resultado final.
    print(f"Test Loss: {average_test_loss:.4f}")
    # Loga a métrica de teste no MLflow.
    mlflow.log_metric("test_loss", average_test_loss)

# Ponto de entrada: executa treinamento e avaliação.
train_model()
