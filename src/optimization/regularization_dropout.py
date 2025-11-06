# Importa utilitário de sistema operacional (não usado diretamente aqui, mas comum em scripts)
import os
# Importa Path para manipulação de caminhos de forma cross-platform
from pathlib import Path
# Importa o núcleo do PyTorch (tensores e operações)
import torch
# Importa módulos de camadas/containers de rede neural
import torch.nn as nn
# Importa o otimizador Adam para atualização de pesos
from torch.optim import Adam
# Importa datasets e transforms do torchvision (ImageFolder, augmentações)
from torchvision import datasets, transforms
# MLflow para experiment tracking (métricas/artefatos)
import mlflow
# Submódulo do MLflow para serializar modelos PyTorch
import mlflow.pytorch
# Importa o gerador pseudoaleatório padrão do Python
import random
# Importa NumPy para utilidades e seeds
import numpy as np

# Define semente global para reprodutibilidade básica
SEED = 42
# Fixa a semente do Python
random.seed(SEED)
# Fixa a semente do NumPy
np.random.seed(SEED)
# Fixa a semente do PyTorch (CPU)
torch.manual_seed(SEED)
# Define o device: aqui fixado em CPU (mude para "cuda" se desejar)
device =  "cpu"
# Se GPU for usada, fixa sementes das GPUs também
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Dimensão lateral das imagens (redimensionadas para 64x64)
image_size = 64                 # 64x64
# Número de canais (3 = RGB)
image_channels = 3              # RGB
# Dimensão do vetor de ruído de entrada do Gerador
g_input_size = 100              # ruído
# Tamanho base do hidden do Gerador
g_hidden_size = 256
# Tamanho da saída do Gerador (imagem flatten: C * H * W)
g_output_size = image_channels * image_size * image_size  # saída flatten
# Tamanho base do hidden do Discriminador
d_hidden_size = 256
# Saída do Discriminador (1 logit por amostra)
d_output_size = 1
# Tamanho do minibatch
batch_size = 64
# Número de épocas de treinamento
num_epochs = 50
# Taxa de aprendizado para Adam
learning_rate = 2e-4
# Suavização de rótulo para “reais” (0.9 ajuda a estabilidade); falsos = 0.0
label_smoothing = 0.9           # opcional: 0.9 para reais; 0.0 para falsos

# Define o diretório base de imagens (usando Path para portabilidade)
data_dir = Path.cwd() / "src" / "optimization" / "imagens"

# Define a cadeia de transforms: resize, tensor e normalização [-1, 1]
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*image_channels, [0.5]*image_channels),
])

# Cria ImageFolder para TRAIN a partir de data_dir/train
train_dataset = datasets.ImageFolder(root=str(data_dir / "train"), transform=transform)
# Cria ImageFolder para VAL a partir de data_dir/val
val_dataset   = datasets.ImageFolder(root=str(data_dir / "val"),   transform=transform)

# Ativa pin_memory apenas se o device for CUDA (cópias H->D mais rápidas)
pin_mem = device == "cuda"
# DataLoader de treino com embaralhamento e drop_last para batches completos
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True,
    pin_memory=pin_mem, num_workers=2
)
# DataLoader de validação sem embaralhar e drop_last para batches completos
val_loader = torch.utils.data.DataLoader(
    val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True,
    pin_memory=pin_mem, num_workers=2
)

# Define o Gerador como um MLP profundo que mapeia ruído -> imagem flatten
class Generator(nn.Module):
    # Construtor recebe dimensões e hiperparâmetros (função de ativação e dropout)
    def __init__(self, input_size, hidden_size, output_size, activation_fn, dropout_rate):
        super().__init__()
        # Empilha camadas lineares com ativações e dropout; saída com Tanh para [-1, 1]
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size * 2),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, output_size),
            nn.Tanh()  # saída em [-1, 1]
        )

    # Forward: recebe z ~ N(0,1) e devolve imagem flatten normalizada em [-1, 1]
    def forward(self, x):
        return self.model(x)

# Define o Discriminador como um MLP que mapeia imagem flatten -> logit (real/falso)
class Discriminator(nn.Module):
    # Construtor recebe dimensões e hiperparâmetros (função de ativação e dropout)
    def __init__(self, input_size, hidden_size, output_size, activation_fn, dropout_rate):
        super().__init__()
        # Empilha camadas lineares com ativações e dropout; saída sem Sigmoid (logits)
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)  # sem Sigmoid (usaremos BCEWithLogitsLoss)
        )

    # Forward: recebe imagem flatten e retorna logit (maior = mais “real”)
    def forward(self, x):
        return self.model(x)

# Define rotina de treino com variação de ativação, L2 e dropout; loga no MLflow
def train(activation_fn, l2_lambda, dropout_rate, version):
    # Instancia Gerador e Discriminador com os hiperparâmetros correntes
    G = Generator(g_input_size, g_hidden_size, g_output_size, activation_fn, dropout_rate).to(device)
    D = Discriminator(g_output_size, d_hidden_size, d_output_size, activation_fn, dropout_rate).to(device)

    # BCE com logits é mais estável (evita Sigmoid explícita no D)
    criterion = nn.BCEWithLogitsLoss()
    # Otimizadores com betas clássicos para GAN e L2 (weight_decay) opcional
    d_optimizer = Adam(D.parameters(), lr=learning_rate, weight_decay=l2_lambda, betas=(0.5, 0.999))
    g_optimizer = Adam(G.parameters(), lr=learning_rate, weight_decay=l2_lambda, betas=(0.5, 0.999))

    # Nomeia o experimento no MLflow com a versão
    experiment_name = f"GAN Custom Dataset Experiment {version}"
    mlflow.set_experiment(experiment_name)

    # Abre um run com tags/params que documentam a varredura de hiperparâmetros
    with mlflow.start_run(run_name=f"GAN_{activation_fn.__name__}_L2_{l2_lambda}_Dropout_{dropout_rate}_v{version}"):
        # Define tags informativas do run
        mlflow.set_tags({
            "version": version,
            "activation_fn": activation_fn.__name__,
            "l2_lambda": l2_lambda,
            "dropout_rate": dropout_rate,
        })
        # Loga hiperparâmetros fixos do modelo/treino
        mlflow.log_params({
            "g_input_size": g_input_size,
            "g_hidden_size": g_hidden_size,
            "g_output_size": g_output_size,
            "d_hidden_size": d_hidden_size,
            "d_output_size": d_output_size,
            "learning_rate": learning_rate,
            "label_smoothing": label_smoothing,
            "image_size": image_size,
            "image_channels": image_channels,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        })

        # Contador global de passos (para métricas por batch)
        global_step = 0
        # Loop de épocas
        for epoch in range(num_epochs):
            # G e D em modo treino
            G.train(); D.train()
            # Acumuladores de perdas por época
            running_d, running_g = 0.0, 0.0

            # Itera sobre batches do DataLoader de treino
            for i, (images, _) in enumerate(train_loader):
                # Tamanho atual do batch (pode variar por drop_last)
                current_bs = images.size(0)
                # Achata imagens reais para vetor 1D e move para device
                real_imgs = images.view(current_bs, -1).to(device, non_blocking=True)

                # ====== Passo do Discriminador (D) ======
                # Garante modo treino do D
                D.train()
                # Zera gradientes do D
                d_optimizer.zero_grad(set_to_none=True)

                # Alvos “reais” com label smoothing (ex.: 0.9)
                real_targets = torch.full((current_bs, 1), label_smoothing, device=device)  # ex.: 0.9
                # Logits do D para imagens reais
                out_real = D(real_imgs)
                # Perda do D nos reais
                d_loss_real = criterion(out_real, real_targets)

                # Amostra ruído z para gerar imagens falsas
                z = torch.randn(current_bs, g_input_size, device=device)
                # Gera imagens falsas e destaca do grafo para não atualizar G
                fake_imgs = G(z).detach()  # detach para não propagar grad no G
                # Alvos “falsos” = 0.0
                fake_targets = torch.zeros(current_bs, 1, device=device)
                # Logits do D para imagens falsas
                out_fake = D(fake_imgs)
                # Perda do D nos falsos
                d_loss_fake = criterion(out_fake, fake_targets)

                # Perda total do D = reais + falsos
                d_loss = d_loss_real + d_loss_fake
                # Backprop em D
                d_loss.backward()
                # Passo de otimização do D
                d_optimizer.step()

                # ====== Passo do Gerador (G) ======
                # G em modo treino
                G.train()
                # Zera gradientes do G
                g_optimizer.zero_grad(set_to_none=True)

                # Reamostra ruído z para o passo do G
                z = torch.randn(current_bs, g_input_size, device=device)
                # Gera imagens candidatas
                gen_imgs = G(z)
                # Faz o D avaliá-las
                out_gen = D(gen_imgs)
                # Alvos para G “enganar” D (quer que D diga real=1)
                g_targets = torch.ones(current_bs, 1, device=device)  # G quer "enganar" D
                # Perda do G (maximiza prob. de D rotular como real)
                g_loss = criterion(out_gen, g_targets)

                # Backprop em G
                g_loss.backward()
                # Passo de otimização do G
                g_optimizer.step()

                # Acumula perdas por época
                running_d += d_loss.item()
                running_g += g_loss.item()

                # Log por batch no MLflow (opcional, a cada 100 batches)
                if i % 100 == 0:
                    mlflow.log_metric("train_batch_d_loss", d_loss.item(), step=global_step)
                    mlflow.log_metric("train_batch_g_loss", g_loss.item(), step=global_step)
                # Incrementa o step global
                global_step += 1

            # Calcula perdas médias da época
            epoch_d = running_d / len(train_loader)
            epoch_g = running_g / len(train_loader)
            # Exibe resumo no console
            print(f"Epoch [{epoch+1}/{num_epochs}] - d_loss: {epoch_d:.4f} | g_loss: {epoch_g:.4f}")
            # Loga métricas de época no MLflow
            mlflow.log_metric("d_loss", epoch_d, step=epoch)
            mlflow.log_metric("g_loss", epoch_g, step=epoch)

        # Ao final, salva Gerador e Discriminador como artefatos do MLflow
        mlflow.pytorch.log_model(G, f"generator_v{version}")
        mlflow.pytorch.log_model(D, f"discriminator_v{version}")

# Define o conjunto de funções de ativação a testar (evitar Sigmoid/Tanh dentro do D)
activation_functions = [nn.ReLU, nn.LeakyReLU]  # (evitar Sigmoid/Tanh em camadas internas do MLP do D)
# Define a grade de valores de L2 (weight decay) a varrer
l2_lambdas = [0.0, 0.01, 0.1]
# Define as taxas de dropout a varrer
dropout_rates = [0.0, 0.3, 0.5]

# Rotina principal: varre combinações e treina/loga cada uma
def main(version="1.0"):
    # Varre todas as combinações de ativação, L2 e dropout
    for activation_fn in activation_functions:
        for l2_lambda in l2_lambdas:
            for dropout_rate in dropout_rates:
                # Mensagem de status para acompanhar o progresso
                print(f"Running: act={activation_fn.__name__}, L2={l2_lambda}, Dropout={dropout_rate}, v={version}")
                # Dispara o treino para a combinação atual
                train(activation_fn, l2_lambda, dropout_rate, version)

# Ponto de entrada: executa a varredura quando chamado como script
if __name__ == "__main__":
    main("1.0")
