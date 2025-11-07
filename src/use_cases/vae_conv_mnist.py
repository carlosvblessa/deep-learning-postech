# Caminho do arquivo (informativo)
# src/use_cases/vae_conv_mnist.py

# Utilidades do SO e paths
import os
from pathlib import Path
# Núcleo do PyTorch e módulos NN/F/optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# MNIST e transforms (pré-processamento)
from torchvision import datasets, transforms
# DataLoader para batching
from torch.utils.data import DataLoader
# Helpers para salvar grids de imagens
from torchvision.utils import save_image, make_grid

# Hiperparâmetros
# Dimensão latente (2 para visualização; 8/16 acelera convergência)
latent_dim   = 8          # 2 para visualização do espaço, 8/16 para convergir mais rápido
# Tamanho do lote
batch_size   = 128
# Épocas de treinamento
num_epochs   = 30
# Taxa de aprendizado
lr           = 1e-3
# Regularização L2 (weight decay)
weight_decay = 1e-5
# Clipping do gradiente (estabilidade)
grad_clip    = 5.0
# Warmup da KL (β cresce linear 0→1 nesses primeiros epochs)
kl_warmup_ep = 10         # beta cresce 0 -> 1 nesses primeiros epochs

# Seleciona device e fixa seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Dataset (valores em [0,1])
# Mantém pixels em [0,1] (adequado para BCE com logits)
transform = transforms.ToTensor()

# Carrega MNIST de treino e teste (baixa se necessário)
train_dataset = datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# DataLoader de treino (embaralha)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=torch.cuda.is_available()
)
# DataLoader de teste (sem embaralhar)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=torch.cuda.is_available()
)

# Modelo
# Encoder convolucional: x → (μ, logσ²)
class ConvEncoder(nn.Module):
    """
    Entrada:  [B,1,28,28]
    Saída:    mu, logvar com dimensão z
    """
    # Define pilha de convs com BN+ReLU e projeções lineares para μ/logσ²
    def __init__(self, z_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> [B,32,14,14]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> [B,64,7,7]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),# -> [B,128,7,7]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Dimensão flatten após convs
        self.flat_dim = 128 * 7 * 7
        # Cabeças para μ e logσ²
        self.fc_mu     = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)

    # Forward: extrai features, achata e projeta para μ/logσ²
    def forward(self, x):
        h = self.conv(x)              # [B,128,7,7]
        h = h.view(x.size(0), -1)     # [B,128*7*7]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder deconvolucional: z → logits de reconstrução
class ConvDecoder(nn.Module):
    """
    Entrada:  z -> logits de reconstrução [B,1,28,28] (sem sigmoid)
    """
    # Projeta z para mapa [128,7,7] e aplica ConvTranspose2d até [1,28,28]
    def __init__(self, z_dim: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> [B,64,14,14]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [B,32,28,28]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)             # -> [B,1,28,28] (logits)
        )

    # Forward: z → fc → reshape → deconv → logits
    def forward(self, z):
        h = self.fc(z)                        # [B,128*7*7]
        h = h.view(z.size(0), 128, 7, 7)     # [B,128,7,7]
        logits = self.deconv(h)               # [B,1,28,28]
        return logits

# VAE completo: Encoder + reparametrização + Decoder
class VAE(nn.Module):
    # Constrói submódulos de encoder/decoder
    def __init__(self, z_dim: int):
        super().__init__()
        self.encoder = ConvEncoder(z_dim)
        self.decoder = ConvDecoder(z_dim)

    # Reparametrização: z = μ + σ ⊙ ε,  ε ~ N(0,I)
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Forward: retorna logits de reconstrução + parâmetros do latente
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decoder(z)
        return recon_logits, mu, logvar

    # Decodifica z em imagens em [0,1] (aplica sigmoid)
    def decode_to_imgs(self, z):
        logits = self.decoder(z)
        return torch.sigmoid(logits)  # imagens em [0,1]

# Instancia o VAE e move para o device
vae = VAE(latent_dim).to(device)

# Otimizador & Scheduler
# Adam com weight decay; scheduler cosseno para lr ao longo das épocas
optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Loss do VAE (com logits)
# Calcula termos de reconstrução (BCEWithLogits) e KLD; soma feita no batch
def vae_losses(recon_logits, x, mu, logvar):
    # Reconstrução usando BCE com logits (mais estável numericamente)
    recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum")
    # Divergência KL entre q(z|x)=N(μ,σ²) e p(z)=N(0,I)
    kld   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon, kld

# Utilitários de salvamento
# Diretório para artefatos (grids de amostras/reconstruções)
OUT_DIR = Path("outputs/mnist_conv_vae")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Salva amostras aleatórias decodificadas de z ~ N(0,I)
@torch.no_grad()
def save_random_samples(epoch: int, n: int = 20):
    vae.eval()
    z = torch.randn(n, latent_dim, device=device)
    imgs = vae.decode_to_imgs(z).cpu()  # [n,1,28,28]
    grid = make_grid(imgs, nrow=min(10, n))
    save_image(grid, OUT_DIR / f"samples_epoch_{epoch:03d}.png")

# Salva grid de entradas reais e suas reconstruções
@torch.no_grad()
def save_reconstructions(epoch: int):
    vae.eval()
    x, _ = next(iter(test_loader))  # um batch de teste
    x = x.to(device)
    logits, _, _ = vae(x)
    x_hat = torch.sigmoid(logits)

    # Seleciona 16 primeiras amostras
    x_vis    = x[:16].cpu()
    xhat_vis = x_hat[:16].cpu()

    # Gera grids (entrada e saída) e salva
    grid_in  = make_grid(x_vis,    nrow=8)
    grid_out = make_grid(xhat_vis, nrow=8)
    save_image(grid_in,  OUT_DIR / f"recons_input_epoch_{epoch:03d}.png")
    save_image(grid_out, OUT_DIR / f"recons_out_epoch_{epoch:03d}.png")

# Treino
# Loop principal de treinamento do VAE com annealing da KL
def train():
    for epoch in range(1, num_epochs + 1):
        vae.train()
        total_recon = 0.0
        total_kld   = 0.0
        total_loss  = 0.0

        # Annealing linear para o peso da KL: beta ∈ [0,1]
        beta = min(1.0, epoch / kl_warmup_ep)

        # Itera batches do loader de treino
        for x, _ in train_loader:
            x = x.to(device)

            # Zera gradientes
            optimizer.zero_grad(set_to_none=True)
            # Forward VAE
            recon_logits, mu, logvar = vae(x)
            # Calcula termos da perda
            recon, kld = vae_losses(recon_logits, x, mu, logvar)
            # Perda total com KL annealing
            loss = recon + beta * kld

            # Backprop
            loss.backward()
            # Clipping do gradiente (evita explosão)
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            # Atualiza parâmetros
            optimizer.step()

            # Acumula somatórios para médias por época
            total_recon += recon.item()
            total_kld   += kld.item()
            total_loss  += loss.item()

        # Normaliza pelas amostras do dataset de treino
        N = len(train_loader.dataset)
        epoch_recon = total_recon / N
        epoch_kld   = total_kld   / N
        epoch_loss  = total_loss  / N

        # Atualiza scheduler de LR
        scheduler.step()

        # Log textual por época
        print(
            f"Epoch {epoch:02d} | "
            f"loss={epoch_loss:.4f} | recon={epoch_recon:.4f} | kl={epoch_kld:.4f} | "
            f"beta={beta:.2f} | lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Gera amostras aleatórias e reconstrói exemplos do conjunto de teste
        save_random_samples(epoch, n=20)
        save_reconstructions(epoch)

    # Mensagem de término e salva um grid maior
    print("Treinamento concluído!")
    # salva um último grid de amostras
    save_random_samples(num_epochs, n=40)

# Execução
# Ponto de entrada do script
if __name__ == "__main__":
    # Roda o treinamento completo
    train()
    # Indica onde os PNGs foram salvos
    print(f"Imagens salvas em: {OUT_DIR.resolve()}")
