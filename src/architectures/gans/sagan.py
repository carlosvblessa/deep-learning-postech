# Importa o núcleo do PyTorch para tensores e computação numérica
import torch
# Importa módulos de camadas e containers de redes neurais
import torch.nn as nn
# Importa o otimizador Adam para atualização de pesos
from torch.optim import Adam
# Importa o MLflow para rastrear experimentos
import mlflow
# Importa utilitários do MLflow específicos para modelos PyTorch
import mlflow.pytorch
# Importa NumPy para operações numéricas e geração de dados
import numpy as np

# Seleciona automaticamente GPU se disponível; caso contrário, usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define semente do PyTorch para reprodutibilidade
torch.manual_seed(42)
# Define semente do NumPy para reprodutibilidade
np.random.seed(42)

# Dimensão do vetor de ruído de entrada do gerador (z)
g_input_size   = 100      # z
# Número base de canais de feature no gerador/discriminador
g_hidden_size  = 128
# Número de canais na imagem de saída (1 = grayscale)
g_output_ch    = 1        # canais de saída (grayscale)
# Tamanho espacial da imagem (32x32)
img_size       = 32
# Tamanho do minibatch usado no treinamento
batch_size     = 64
# Número total de épocas de treinamento
epochs         = 1000
# Taxa de aprendizado para ambos os otimizadores
lr             = 2e-4

# Função geradora de “dados reais” sintéticos em [-1, 1] com shape [B, 1, 32, 32]
def get_real_data():
    # Amostra normal padrão e organiza no formato de imagem (B,C,H,W) no device atual
    x = torch.randn(batch_size, g_output_ch, img_size, img_size, device=device)
    # Comprime o intervalo para [-1, 1] aplicando Tanh
    return torch.tanh(x)

# Bloco de autoatenção 2D; retorna apenas o tensor (para uso em nn.Sequential)
class SelfAttention(nn.Module):
    # Inicializa camadas 1x1 para query, key e value e o coeficiente residual gamma
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim,      1)
        self.gamma = nn.Parameter(torch.zeros(1))

    # Computa atenção não-local no mapa de features e mistura com o residual
    def forward(self, x):
        # Obtém batch, canais e dimensões espaciais
        B, C, H, W = x.shape
        # Projeta queries e remodela para [B, HW, C/8]
        q = self.query_conv(x).view(B, -1, H*W).transpose(1, 2)   # [B, HW, C/8]
        # Projeta keys e remodela para [B, C/8, HW]
        k = self.key_conv(x).view(B, -1, H*W)                     # [B, C/8, HW]
        # Calcula mapa de atenção como softmax(QK^T) com shape [B, HW, HW]
        attn = torch.softmax(q @ k, dim=-1)                       # [B, HW, HW]
        # Projeta values e remodela para [B, C, HW]
        v = self.value_conv(x).view(B, C, H*W)                    # [B, C, HW]
        # Aplica atenção em V e volta para [B, C, H, W]
        out = (v @ attn.transpose(1, 2)).view(B, C, H, W)         # [B, C, H, W]
        # Retorna combinação residual: x + gamma * atenção
        return self.gamma * out + x

# Define o Gerador no estilo DCGAN com um bloco de Self-Attention em 8x8
class SAGANGenerator(nn.Module):
    # Constrói a pilha transposta (upsampling) até 32x32 + Tanh
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Projeta z: [B, 100, 1, 1] -> [B, 512, 4, 4]
            nn.ConvTranspose2d(g_input_size, g_hidden_size * 4, 4, 1, 0, bias=False),
            # Normaliza feature maps para estabilizar o treino
            nn.BatchNorm2d(g_hidden_size * 4),
            # Ativação ReLU
            nn.ReLU(True),

            # Upsample: [B, 512, 4, 4] -> [B, 256, 8, 8]
            nn.ConvTranspose2d(g_hidden_size * 4, g_hidden_size * 2, 4, 2, 1, bias=False),
            # BatchNorm após upsample
            nn.BatchNorm2d(g_hidden_size * 2),
            # ReLU novamente
            nn.ReLU(True),

            # Inserção de Self-Attention no mapa 8x8 (canal = 256)
            SelfAttention(g_hidden_size * 2),

            # Upsample: [B, 256, 8, 8] -> [B, 128, 16, 16]
            nn.ConvTranspose2d(g_hidden_size * 2, g_hidden_size, 4, 2, 1, bias=False),
            # Normalização em batch
            nn.BatchNorm2d(g_hidden_size),
            # ReLU
            nn.ReLU(True),

            # Upsample final: [B, 128, 16, 16] -> [B, 1, 32, 32]
            nn.ConvTranspose2d(g_hidden_size, g_output_ch, 4, 2, 1, bias=False),
            # Mapeia para [-1, 1] (distribuição de pixels)
            nn.Tanh()
        )

    # Encaminha o vetor latente pelo gerador para obter uma imagem
    def forward(self, z):
        return self.net(z)

# Define o Discriminador com Spectral Normalization e Self-Attention em 8x8
class SAGANDiscriminator(nn.Module):
    # Constrói a pilha de convoluções descendentes até um logit escalar
    def __init__(self):
        super().__init__()
        # Atalho para aplicar spectral normalization nas camadas
        sn = nn.utils.spectral_norm  # SAGAN usa SN no D
        self.net = nn.Sequential(
            # Downsample: [B, 1, 32, 32] -> [B, 128, 16, 16]
            sn(nn.Conv2d(g_output_ch, g_hidden_size, 4, 2, 1, bias=False)),   # 32->16
            # Ativação LeakyReLU mais estável para D
            nn.LeakyReLU(0.2, inplace=True),

            # Downsample: [B, 128, 16, 16] -> [B, 256, 8, 8]
            sn(nn.Conv2d(g_hidden_size, g_hidden_size * 2, 4, 2, 1, bias=False)),  # 16->8
            # LeakyReLU
            nn.LeakyReLU(0.2, inplace=True),

            # Self-Attention no nível 8x8 (canal = 256)
            SelfAttention(g_hidden_size * 2),  # atenção no 8x8

            # Downsample: [B, 256, 8, 8] -> [B, 512, 4, 4]
            sn(nn.Conv2d(g_hidden_size * 2, g_hidden_size * 4, 4, 2, 1, bias=False)),  # 8->4
            # LeakyReLU
            nn.LeakyReLU(0.2, inplace=True),

            # Projeção final para 1x1 (logits): [B, 512, 4, 4] -> [B, 1, 1, 1]
            sn(nn.Conv2d(g_hidden_size * 4, 1, 4, 1, 0, bias=False))  # 4->1 (logits)
        )

    # Produz um logit [B,1] para cada imagem de entrada
    def forward(self, x):
        return self.net(x).view(x.size(0), 1)  # [B, 1] logits

# Laço principal de treinamento da GAN (alternando D e G)
def train():
    # Instancia gerador e discriminador no device adequado
    G = SAGANGenerator().to(device)
    D = SAGANDiscriminator().to(device)
    # Usa BCE com logits (dispensa Sigmoid na saída do D)
    criterion = nn.BCEWithLogitsLoss()  # mais estável que Sigmoid+BCELoss
    # Otimizador de D com betas típicos para GAN
    d_opt = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # Otimizador de G com mesmos hiperparâmetros
    g_opt = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    # Define o experimento no MLflow para registro de métricas/artefatos
    mlflow.set_experiment("SAGAN Training (simplified 32x32)")
    # Inicia uma execução de tracking no MLflow
    with mlflow.start_run():
        # Loop de épocas
        for epoch in range(epochs):
            # ======== Passo do Discriminador (maximiza separação real/falso) ========
            # Amostra um batch de “reais” sintéticos
            real = get_real_data()
            # Amostra um batch de ruído z ~ N(0,1) em formato [B, z, 1, 1]
            z = torch.randn(batch_size, g_input_size, 1, 1, device=device)
            # Gera “falsos” e desconecta o gradiente para não atualizar G aqui
            fake = G(z).detach()

            # Zera gradientes do D
            d_opt.zero_grad()
            # Avalia logits para reais
            d_real = D(real)
            # Avalia logits para falsos
            d_fake = D(fake)
            # Perda do D para reais (alvo = 1)
            d_real_loss = criterion(d_real, torch.ones(batch_size, 1, device=device))
            # Perda do D para falsos (alvo = 0)
            d_fake_loss = criterion(d_fake, torch.zeros(batch_size, 1, device=device))
            # Soma total da perda do D
            d_loss = d_real_loss + d_fake_loss
            # Backpropaga a perda e atualiza D
            d_loss.backward()
            d_opt.step()

            # ======== Passo do Gerador (engana D fazendo-o prever 1) ========
            # Novo ruído para gerar outro batch
            z = torch.randn(batch_size, g_input_size, 1, 1, device=device)
            # Gera imagens
            gen = G(z)
            # Zera gradientes do G
            g_opt.zero_grad()
            # Passa imagens geradas pelo D para obter logits
            d_gen = D(gen)
            # Objetivo do G: que D classifique como real (alvo = 1)
            g_loss = criterion(d_gen, torch.ones(batch_size, 1, device=device))
            # Backpropaga a perda e atualiza G
            g_loss.backward()
            g_opt.step()

            # A cada 100 épocas, imprime e loga métricas no MLflow
            if epoch % 100 == 0:
                print(f"Epoch {epoch:04d} | D: R {d_real_loss.item():.3f} F {d_fake_loss.item():.3f} "
                      f"T {d_loss.item():.3f} | G: {g_loss.item():.3f}")
                mlflow.log_metric("d_real_loss", d_real_loss.item(), step=epoch)
                mlflow.log_metric("d_fake_loss", d_fake_loss.item(), step=epoch)
                mlflow.log_metric("d_loss", d_loss.item(), step=epoch)
                mlflow.log_metric("g_loss", g_loss.item(), step=epoch)

        # Ao final, registra hiperparâmetros importantes do experimento
        mlflow.log_params({
            "g_input_size": g_input_size,
            "g_hidden_size": g_hidden_size,
            "g_output_ch": g_output_ch,
            "img_size": img_size,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
        })
        # Salva os modelos (gerador e discriminador) como artefatos no MLflow
        mlflow.pytorch.log_model(G, "generator")
        mlflow.pytorch.log_model(D, "discriminator")

# Ponto de entrada do script: executa o treinamento quando chamado diretamente
if __name__ == "__main__":
    train()
