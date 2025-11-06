# Importa o núcleo do PyTorch para tensores e computação numérica
import torch
# Importa camadas/containers de rede neural
import torch.nn as nn
# Funções utilitárias (one_hot, etc.) e ativação/loss sem camadas
import torch.nn.functional as F
# Otimizador Adam para atualização dos pesos
from torch.optim import Adam
# MLflow para experiment tracking
import mlflow
# Submódulo do MLflow para serializar modelos PyTorch
import mlflow.pytorch
# Biblioteca NumPy para manipulação de arrays e geração de dados
import numpy as np

# Seleciona GPU se disponível; caso contrário, usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Fixa a semente para reprodutibilidade básica
torch.manual_seed(42)
np.random.seed(42)

# Desvio-padrão da distribuição alvo (dados reais)
data_stddev = 1.25

# Dimensão do ruído de entrada do gerador
g_input_size = 50
# Tamanho do hidden do gerador (capacidade/complexidade)
g_hidden_size = 128
# Saída escalar do gerador (gera um valor 1D)
g_output_size = 1
# Tamanho do hidden do discriminador
d_hidden_size = 128
# Número de classes (condição categórica)
label_size = 10
# Tamanho do minibatch usado no treino
minibatch_size = 100
# Número de épocas de treinamento
num_epochs = 1000
# Taxa de aprendizado para Adam (G e D)
lr = 2e-4

# Gera um minibatch de dados reais condicionais em rótulos (classe -> média)
def get_real_data(batch_size=minibatch_size):
    # Amostra rótulos inteiros uniformemente entre 0 e label_size-1
    labels = torch.randint(0, label_size, (batch_size,), device=device)
    # Define uma média por classe (ex.: intervalo ~[4,6] igualmente espaçado)
    means = 4.0 + (labels.float() / (label_size - 1)) * 2.0  # ~[4,6]
    # Amostra dados 1D ~ N(means, data_stddev^2) e mantém shape [B, 1]
    x = torch.normal(means, data_stddev).unsqueeze(1)        # [B, 1]
    # Retorna pares (x, labels inteiros)
    return x, labels

# Converte rótulos inteiros para one-hot float (necessário para condicionar)
def one_hot(labels, num_classes):
    # Retorna tensor [B, num_classes] com 0/1 em cada categoria
    return F.one_hot(labels, num_classes=num_classes).float()

# Define o Gerador (condicional) como MLP simples
class Generator(nn.Module):
    # Construtor: empilha camadas lineares + ReLU
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_input_size + label_size, g_hidden_size),
            nn.ReLU(True),
            nn.Linear(g_hidden_size, g_hidden_size),
            nn.ReLU(True),
            nn.Linear(g_hidden_size, g_output_size)  # logits reais (sem Tanh/Sigmoid)
        )
    # Forward: concatena ruído z e one-hot y e passa no MLP
    def forward(self, z, y_oh):
        # Concatenação na dimensão de features (B, z+y)
        x = torch.cat([z, y_oh], dim=1)
        # Produz escalar gerado por amostra
        return self.net(x)

# Define o Discriminador (condicional) como MLP
class Discriminator(nn.Module):
    # Construtor: MLP com LeakyReLU e saída 1 (logits)
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_output_size + label_size, d_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_hidden_size, d_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_hidden_size, 1)  # logits (usar BCEWithLogitsLoss)
        )
    # Forward: concatena x real/falso e y_onehot e classifica como real/falso
    def forward(self, x, y_oh):
        # Concatenação na dimensão de features (B, 1+y)
        h = torch.cat([x, y_oh], dim=1)
        # Retorna logits (sem Sigmoid)
        return self.net(h)

# Laço principal de treinamento (configura MLflow, otimiza D e G)
def train():
    # Instancia gerador e discriminador no device escolhido
    G = Generator().to(device)
    D = Discriminator().to(device)
    # Loss binária estável para logits (combina Sigmoid internamente)
    criterion = nn.BCEWithLogitsLoss()
    # Otimizadores de G e D, com betas clássicos para GAN
    d_opt = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_opt = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    # Nomeia/seleciona o experimento no MLflow
    mlflow.set_experiment("CGAN Training (1D)")
    # Abre um run no MLflow para registrar métricas/artefatos
    with mlflow.start_run():
        # Loop de épocas
        for epoch in range(num_epochs):
            # Coloca os modelos em modo treino (dropout/bn se houvesse)
            D.train(); G.train()

            # Atualização do Discriminador (passo de D)
            # Amostra batch real e rótulos inteiros
            real_x, real_y = get_real_data(minibatch_size)
            # Converte rótulos para one-hot
            real_y_oh = one_hot(real_y, label_size)

            # Amostra ruído z e rótulos para dados falsos
            z = torch.randn(minibatch_size, g_input_size, device=device)
            fake_y = torch.randint(0, label_size, (minibatch_size,), device=device)
            fake_y_oh = one_hot(fake_y, label_size)
            # Gera falsos e destaca do grafo (não atualiza G no passo de D)
            fake_x = G(z, fake_y_oh).detach()

            # Zera gradientes de D
            d_opt.zero_grad()
            # Logits para reais (x,y) e falsos (x~,y)
            real_logits = D(real_x, real_y_oh)
            fake_logits = D(fake_x, fake_y_oh)
            # Loss de D nos reais (targets=1)
            d_real_loss = criterion(real_logits, torch.ones(minibatch_size, 1, device=device))
            # Loss de D nos falsos (targets=0)
            d_fake_loss = criterion(fake_logits, torch.zeros(minibatch_size, 1, device=device))
            # Loss total de D = reais + falsos
            d_loss = d_real_loss + d_fake_loss
            # Backprop apenas em D
            d_loss.backward()
            # Passo de otimização do D
            d_opt.step()

            # Atualização do Gerador (passo de G)
            # Reamostra ruído e rótulos (opcional, ajuda na diversidade)
            z = torch.randn(minibatch_size, g_input_size, device=device)
            fake_y = torch.randint(0, label_size, (minibatch_size,), device=device)
            fake_y_oh = one_hot(fake_y, label_size)

            # Zera gradientes de G
            g_opt.zero_grad()
            # Gera exemplos falsos condicionalmente
            gen_x = G(z, fake_y_oh)
            # Pede para D “acreditar” que são reais (targets=1)
            gen_logits = D(gen_x, fake_y_oh)
            # Loss de G (engana D)
            g_loss = criterion(gen_logits, torch.ones(minibatch_size, 1, device=device))
            # Backprop em G
            g_loss.backward()
            # Passo de otimização do G
            g_opt.step()

            # Logging/print a cada 100 épocas para acompanhar o treino
            if epoch % 100 == 0:
                # Feedback no console com as perdas parciais
                print(f"Epoch {epoch:04d} | D: real {d_real_loss.item():.3f} "
                      f"fake {d_fake_loss.item():.3f} tot {d_loss.item():.3f} | "
                      f"G: {g_loss.item():.3f}")
                # Registra métricas no MLflow com a etapa (step) = época
                mlflow.log_metric("d_real_loss", d_real_loss.item(), step=epoch)
                mlflow.log_metric("d_fake_loss", d_fake_loss.item(), step=epoch)
                mlflow.log_metric("d_loss", d_loss.item(), step=epoch)
                mlflow.log_metric("g_loss", g_loss.item(), step=epoch)

        # Ao final do treino, loga hiperparâmetros usados
        mlflow.log_params({
            "g_input_size": g_input_size,
            "g_hidden_size": g_hidden_size,
            "g_output_size": g_output_size,
            "d_hidden_size": d_hidden_size,
            "label_size": label_size,
            "minibatch_size": minibatch_size,
            "lr": lr,
            "epochs": num_epochs,
        })
        # Salva os artefatos de modelo (G e D) no MLflow
        mlflow.pytorch.log_model(G, "generator")
        mlflow.pytorch.log_model(D, "discriminator")

# Ponto de entrada: executa o treino quando rodado como script
if __name__ == "__main__":
    train()
