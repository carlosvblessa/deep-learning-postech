# Caminho do arquivo (informativo)
# src/use_cases/vae.py

# Importações padrão
import os
from pathlib import Path
# Aleatoriedade controlada (gera dados sintéticos reproduzíveis)
import random
# NumPy para utilidades numéricas
import numpy as np
# Núcleo do PyTorch (tensores)
import torch
# Módulos de rede neural do PyTorch
import torch.nn as nn
# Otimizadores do PyTorch
import torch.optim as optim
# Datasets (MNIST) e transforms (pré-processamento)
from torchvision import datasets, transforms
# DataLoader para batching/iteração em mini-batches
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
# Matplotlib para visualizar amostras geradas
import matplotlib.pyplot as plt

# Tamanho lateral das imagens (MNIST é 28x28)
image_size = 28          # MNIST: 28x28
# Número de canais (MNIST é escala de cinza)
image_channels = 1       # MNIST é grayscale
# Dimensão do espaço latente (2D facilita plot/visualização)
latent_dim = 2           # Espaço latente 2D (bom p/ visualizar)
# Tamanho do batch para treino
batch_size = 128
# Número de épocas de treinamento
num_epochs = 50
# Taxa de aprendizado do otimizador
learning_rate = 1e-3

# Seed fixa para reprodutibilidade
SEED = 42
# Fixa a seed do gerador de números aleatórios do Python
random.seed(SEED)
# Fixa a seed do NumPy
np.random.seed(SEED)
# Fixa a seed do PyTorch (CPU)
torch.manual_seed(SEED)
# Ajustes de reproducibilidade quando há CUDA disponível
if torch.cuda.is_available():
    # Fixa a seed de todos os dispositivos CUDA
    torch.cuda.manual_seed_all(SEED)
    # Deixa o backend determinístico (pode reduzir performance)
    torch.backends.cudnn.deterministic = True
    # Desativa heurísticas de benchmark do cuDNN
    torch.backends.cudnn.benchmark = False

# Seleciona GPU se disponível; caso contrário, usa CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Pipeline de pré-processamento
# Mantém pixels em [0,1] (sem normalizar para [-1,1]) para usar BCE + sigmoid
transform = transforms.Compose([
    # Converte PIL->Tensor e escala para [0,1]
    transforms.ToTensor()  # produz tensores em [0,1]
    # NÃO normalizar para [-1,1] quando usar BCE com saída sigmoid
])

# Carrega MNIST de treino (baixando se não existir)
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# Determina número de workers de I/O (fallback seguro se 'os' não existir)
num_workers = min(4, (os.cpu_count() or 1)) if hasattr(__import__('os'), 'cpu_count') else 0
# Usa pin_memory se houver CUDA (otimiza H2D)
pin_memory = torch.cuda.is_available()

# Cria DataLoader de treino (embaralha amostras)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0)
)

# Encoder: mapeia imagem flatten -> (mu, logvar)
class Encoder(nn.Module):
    # Define camadas lineares para extrair features e inferir parâmetros da Gaussiana
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Projeção inicial para espaço escondido
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Cabeça para média (μ) do latente
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # Cabeça para log-variância (log σ²) do latente
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    # Forward do encoder: retorna parâmetros (μ, log σ²)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder: mapeia z (latente) -> imagem reconstruída
class Decoder(nn.Module):
    # Duas camadas lineares com ReLU + Sigmoid para reconstrução em [0,1]
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        # Projeção do latente para espaço escondido
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        # Projeção final para dimensão da imagem flatten
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    # Forward do decoder: gera x̂ no intervalo [0,1]
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))  # saída em [0,1]
        return x_reconstructed

# VAE: junta Encoder + Decoder + reparametrização
class VAE(nn.Module):
    # Recebe instâncias de encoder/decoder
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # Trucagem de reparametrização: z = μ + σ ⊙ ε, com ε ~ N(0, I)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Forward do VAE: retorna reconstrução e parâmetros (μ, log σ²)
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# Função de perda do VAE (ELBO negativa ≈ BCE + KLD)
def loss_function(x_reconstructed, x, mu, logvar):
    # BCE somada no batch (reconstrução); alvos x em [0,1]
    BCE = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    # KLD entre N(μ, σ²) e N(0, I): força latente a aproximar prior padrão
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Perda total (minimizar)
    return BCE + KLD

# Dimensão de entrada flatten (28*28)
input_dim = image_size * image_size
# Tamanho da camada escondida MLP
hidden_dim = 256
# Dimensão de saída do decoder (igual à entrada flatten)
output_dim = input_dim

# Instancia encoder com (784 → 256 → μ/logvar)
encoder = Encoder(input_dim, hidden_dim, latent_dim)
# Instancia decoder com (z → 256 → 784)
decoder = Decoder(latent_dim, hidden_dim, output_dim)
# Cria VAE agregando encoder/decoder e move para device
vae = VAE(encoder, decoder).to(device)

# Otimizador Adam nos parâmetros do VAE
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Loop de treinamento do VAE
def train_vae():
    # Modo treino (habilita grad)
    vae.train()
    # Número total de amostras (para média de perda por amostra)
    dataset_size = len(train_loader.dataset)
    # Loop de épocas
    for epoch in range(num_epochs):
        # Acumulador de perda total da época
        train_loss = 0.0
        # Itera lotes do DataLoader
        for data, _ in train_loader:
            # Flatten: [B, 1, 28, 28] -> [B, 784] e move para device
            data = data.view(-1, input_dim).to(device)  # [B, 784]

            # Zera gradientes anteriores
            optimizer.zero_grad()
            # Forward completo: reconstrução + parâmetros
            x_reconstructed, mu, logvar = vae(data)
            # Calcula ELBO (BCE + KLD)
            loss = loss_function(x_reconstructed, data, mu, logvar)
            # Backprop da perda
            loss.backward()
            # Atualiza pesos
            optimizer.step()

            # Acumula perda (somada) do batch
            train_loss += loss.item()

        # Perda média por amostra na época
        avg_loss = train_loss / dataset_size
        # Log de acompanhamento
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # Mensagem final de conclusão
    print("Treinamento concluído!")

def generate_images(num_images=10, out_dir="generated", save_grid=True, grid_cols=None):
    """
    Gera imagens do VAE e salva em disco.
    - num_images: número de amostras a gerar
    - out_dir: diretório de saída
    - save_grid: também salva uma grade com todas as amostras
    - grid_cols: nº de colunas da grade (auto se None)
    Retorna dict com caminhos dos arquivos gerados.
    """
    vae.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        # Tensores em [0,1]; formato [B, 1, 28, 28]
        generated = vae.decoder(z).cpu().view(-1, 1, image_size, image_size)

    # Salva individuais
    sample_paths = []
    for i in range(num_images):
        fp = out_dir / f"sample_{i+1:03d}.png"
        # save_image aceita [C, H, W] com valores em [0,1]
        save_image(generated[i], fp)
        sample_paths.append(str(fp))

    # Salva grade (opcional)
    grid_path = None
    if save_grid:
        if grid_cols is None:
            grid_cols = min(10, num_images)
        grid = make_grid(generated, nrow=grid_cols)  # [C, H, W]
        grid_path = out_dir / f"grid_{num_images}.png"
        save_image(grid, grid_path)

    return {"samples": sample_paths, "grid": (str(grid_path) if grid_path else None)}

# Ponto de entrada do script
if __name__ == "__main__":
    train_vae()
    result = generate_images(num_images=20, out_dir="outputs/mnist_vae")
    print(f"Salvos {len(result['samples'])} PNGs em {Path(result['samples'][0]).parent}")
    if result["grid"]:
        print(f"Grid salvo em: {result['grid']}")
