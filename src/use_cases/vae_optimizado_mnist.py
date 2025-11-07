# Caminho do arquivo (informativo)
# src/use_cases/vae_optimizado_mnist.py

# Utilidades do SO e paths (para salvar saídas)
import os
from pathlib import Path
# Funções matemáticas auxiliares (pode ser útil para análises)
import math
# Núcleo do PyTorch e módulos de NN/optim/F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# MNIST e transforms (pré-processamento)
from torchvision import datasets, transforms
# DataLoader para batching/shuffling
from torch.utils.data import DataLoader
# Utilitários para salvar imagens e grids
from torchvision.utils import save_image, make_grid

# -----------------------
# Hiperparâmetros
# -----------------------
# Lado da imagem (MNIST 28x28)
image_size   = 28
# Dimensão de entrada flatten (28*28)
input_dim    = image_size * image_size
# Número de canais (MNIST em tons de cinza)
image_ch     = 1
# Dimensão do espaço latente (2 p/ visualizar; 8 acelera convergência)
latent_dim   = 2          # experimente 8 para queda mais rápida
# Larguras das camadas ocultas do MLP
hidden1      = 512
hidden2      = 256
# Tamanho do lote
batch_size   = 128
# Número de épocas
num_epochs   = 30
# Taxa de aprendizado
lr           = 1e-3
# Warmup da KL: beta cresce linear 0→1 nos primeiros epochs
kl_warmup_ep = 10         # epochs para beta ir de 0->1
# Regularização L2 (weight decay) no otimizador
weight_decay = 1e-5
# Clipping de gradiente (estabilidade numérica)
grad_clip    = 5.0

# Seleciona device e fixa seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed determinística básica
torch.manual_seed(42)
# Seed adicional para CUDA (se houver)
torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None

# -----------------------
# Dataset (sem Normalize)
# -----------------------
# Mantém pixels em [0,1] — adequado para BCE com logits
transform = transforms.Compose([
    transforms.ToTensor(),  # valores já em [0,1]
])

# Baixa/abre MNIST de treino
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# DataLoader de treino com embaralhamento
train_loader  = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

# -----------------------
# Modelo
# -----------------------
# Encoder MLP: x_flat → (μ, logσ²)
class Encoder(nn.Module):
    # Define camadas lineares com ReLU e cabeças para μ/logσ²
    def __init__(self, in_dim, h1, h2, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(h2, z_dim)
        self.fc_logvar = nn.Linear(h2, z_dim)

    # Forward: extrai features e projeta para μ/logσ²
    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder MLP: z → logits da reconstrução (sem sigmoid)
class Decoder(nn.Module):
    # Define projeções de z até dimensão de saída (784)
    def __init__(self, z_dim, h2, h1, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, out_dim)   # LOGITS (sem sigmoid aqui)
        )

    # Forward: produz logits de reconstrução
    def forward(self, z):
        logits = self.net(z)         # [B, 784] logits
        return logits

# VAE completo: Encoder + reparametrização + Decoder
class VAE(nn.Module):
    # Recebe encoder e decoder prontos (injeção de dependência)
    def __init__(self, enc: Encoder, dec: Decoder):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    # Reparametrização: z = μ + σ ⊙ ε, com ε ~ N(0, I)
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Forward: devolve logits de reconstrução + parâmetros latentes
    def forward(self, x_flat):
        mu, logvar = self.encoder(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decoder(z)
        return recon_logits, mu, logvar

    # Decodifica z para imagens em [0,1] (aplica sigmoid nos logits)
    def decode_to_imgs(self, z):
        """Para geração (aplica sigmoid nos logits)."""
        logits = self.decoder(z)
        x_hat = torch.sigmoid(logits)
        return x_hat.view(-1, 1, image_size, image_size)

# Instancia encoder/decoder e o VAE; move para o device
encoder = Encoder(input_dim, hidden1, hidden2, latent_dim)
decoder = Decoder(latent_dim, hidden2, hidden1, input_dim)
vae = VAE(encoder, decoder).to(device)

# -----------------------
# Otimizador & Scheduler
# -----------------------
# Adam com weight decay moderado (regularização)
optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
# Scheduler cosseno para variar LR ao longo das épocas
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# -----------------------
# Loss do VAE (com logits)
# -----------------------
# Calcula perdas de reconstrução (BCEWithLogits) e KLD
def vae_losses(recon_logits, x_flat, mu, logvar):
    # Reconstrução: BCE com logits — estável e apropriada para [0,1]
    # reduction='sum' aproxima a NLL canônica em VAEs
    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, x_flat, reduction='sum')
    # KLD entre q(z|x)=N(μ,σ²) e p(z)=N(0,I)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld

# -----------------------
# Treino
# -----------------------
# Laço principal de treinamento com KL warmup e clipping
def train():
    vae.train()
    for epoch in range(1, num_epochs + 1):
        total_recon = 0.0
        total_kld   = 0.0
        total_loss  = 0.0

        # Peso da KL (beta) com annealing linear 0→1
        beta = min(1.0, epoch / kl_warmup_ep)

        # Itera mini-batches do MNIST
        for data, _ in train_loader:
            data = data.to(device)                  # [B, 1, 28, 28] em [0,1]
            x = data.view(-1, input_dim)            # [B, 784]

            # Zera gradientes (set_to_none otimiza memória)
            optimizer.zero_grad(set_to_none=True)

            # Forward no VAE
            recon_logits, mu, logvar = vae(x)
            # Calcula termos da perda
            recon, kld = vae_losses(recon_logits, x, mu, logvar)
            # Combina perdas (ELBO negativa): recon + β*KL
            loss = recon + beta * kld

            # Backprop + atualização
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            optimizer.step()

            # Acumula somatórios para médias por época
            total_recon += recon.item()
            total_kld   += kld.item()
            total_loss  += loss.item()

        # Normaliza por número de exemplos do dataset
        N = len(train_loader.dataset)
        epoch_recon = total_recon / N
        epoch_kld   = total_kld   / N
        epoch_loss  = total_loss  / N

        # Avança o scheduler (ajusta LR)
        scheduler.step()

        # Log textual do progresso
        print(
            f"Epoch {epoch:02d} | "
            f"loss={epoch_loss:.4f} | recon={epoch_recon:.4f} | kl={epoch_kld:.4f} | "
            f"beta={beta:.2f} | lr={scheduler.get_last_lr()[0]:.6f}"
        )

    # Mensagem final após completar as épocas
    print("Treinamento concluído!")

# -----------------------
# Geração e salvamento
# -----------------------
# Gera amostras do decodificador e salva PNGs individuais e em grid
def generate_and_save(num_images=20, out_dir="outputs/mnist_vae"):
    vae.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Amostra z ~ N(0,I) e decodifica para imagens
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        imgs = vae.decode_to_imgs(z).cpu()  # [B,1,28,28], em [0,1]

    # Salva imagens individuais
    for i in range(num_images):
        save_image(imgs[i], out / f"sample_{i+1:03d}.png")

    # Salva um grid com todas as amostras
    grid = make_grid(imgs, nrow=min(10, num_images))
    save_image(grid, out / f"grid_{num_images}.png")

    # Feedback do caminho de saída
    print(f"Salvos {num_images} PNGs em {out}")

# -----------------------
# Execução
# -----------------------
# Ponto de entrada do script
if __name__ == "__main__":
    # Treina o VAE com KL warmup
    train()
    # Gera amostras e salva em disco
    generate_and_save(num_images=20)
