# Importa o núcleo do PyTorch para tensores e computação numérica
import torch
# Importa camadas/containers de rede neural
import torch.nn as nn
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
# Fixa a semente do PyTorch para reprodutibilidade
torch.manual_seed(42)
# Fixa a semente do NumPy para reprodutibilidade
np.random.seed(42)

# Dimensão do vetor latente (ruído) de entrada do gerador
latent_dim = 128
# Menor resolução inicial (4x4)
start_resolution = 4
# Maior resolução alvo (64x64)
max_resolution   = 64
# Número de épocas por estágio de resolução
epochs_per_stage = 100
# Tamanho do minibatch
batch_size       = 64
# Taxa de aprendizado dos otimizadores
lr = 2e-4

# Lista que conterá as resoluções progressivas (4,8,16,32,64)
RES_LIST = []
# Começa em 4 e dobra até atingir o máximo
r = start_resolution
# Constrói a pirâmide de resoluções (progressive growing)
while r <= max_resolution:
    RES_LIST.append(r)
    r *= 2
# Dicionário com o número de canais por resolução (pode ajustar conforme hardware/dados)
CHAN = {4: 256, 8: 128, 16: 64, 32: 32, 64: 16}  # ajuste conforme GPU/dados

# Define o gerador da ProGAN simplificada
class Generator(nn.Module):
    # Construtor recebe a dimensão latente (ruído)
    def __init__(self, latent_dim=128):
        # Inicializa a superclasse nn.Module
        super().__init__()
        # Armazena a dimensão latente
        self.latent_dim = latent_dim

        # Bloco inicial: projeta z (latente) para um mapa 4x4 com CHAN[4] canais
        self.const_block = nn.Sequential(
            # Deconvolução para obter um mapa 4x4 a partir de (C=latent_dim,1,1)
            nn.ConvTranspose2d(latent_dim, CHAN[4], kernel_size=4, stride=1, padding=0),  # -> 4x4
            # Normalização em batch para estabilizar o treinamento
            nn.BatchNorm2d(CHAN[4]),
            # Ativação não-linear LeakyReLU
            nn.LeakyReLU(0.2, inplace=True),
            # Convolução 3x3 para refinar as features em 4x4
            nn.Conv2d(CHAN[4], CHAN[4], kernel_size=3, padding=1),
            # Normalização em batch após a convolução
            nn.BatchNorm2d(CHAN[4]),
            # Outra não-linearidade
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Dicionário de blocos progressivos (upsample x2 a cada estágio)
        self.blocks = nn.ModuleDict()
        # Camadas de saída para converter feature maps em imagem (1 canal) por resolução
        self.toRGB  = nn.ModuleDict()
        # Percorre todas as resoluções configuradas
        for i, res in enumerate(RES_LIST):
            # Para a resolução base 4x4, apenas define o toRGB
            if res == 4:
                # Projeta de CHAN[4] -> 1 canal (grayscale); pode trocar p/ 3 canais se quiser RGB
                self.toRGB[str(res)] = nn.Conv2d(CHAN[4], 1, kernel_size=1)  # 4x4 -> 1ch
            else:
                # Resolução anterior (metade)
                prev = RES_LIST[i-1]
                # Bloco de upsample: sobe de prev -> res (dobrando H e W)
                self.blocks[str(res)] = nn.Sequential(
                    # Transposed conv para upsample 2x (stride=2)
                    nn.ConvTranspose2d(CHAN[prev], CHAN[res], kernel_size=4, stride=2, padding=1),  # upsample x2
                    # BatchNorm após upsample
                    nn.BatchNorm2d(CHAN[res]),
                    # Ativação
                    nn.LeakyReLU(0.2, inplace=True),
                    # Convolução 3x3 para refinar
                    nn.Conv2d(CHAN[res], CHAN[res], kernel_size=3, padding=1),
                    # BatchNorm novamente
                    nn.BatchNorm2d(CHAN[res]),
                    # Ativação
                    nn.LeakyReLU(0.2, inplace=True),
                )
                # toRGB correspondente a esta resolução
                self.toRGB[str(res)] = nn.Conv2d(CHAN[res], 1, kernel_size=1)

    # Define o fluxo forward do gerador
    def forward(self, z, stage: int):
        """
        z: [B, latent_dim, 1, 1]
        stage: 0 -> 4x4, 1 -> 8x8, ..., len(RES_LIST)-1 -> max_resolution
        """
        # Projeta o ruído para features 4x4
        x = self.const_block(z)  # 4x4
        # Para cada estágio além do 4x4, aplica o bloco de upsample correspondente
        for i in range(1, stage + 1):
            res = RES_LIST[i]
            x = self.blocks[str(res)](x)
        # Seleciona a resolução atual e projeta para imagem (sem Tanh — usaremos BCEWithLogits no D)
        res_now = RES_LIST[stage]
        return self.toRGB[str(res_now)](x)


# Define o discriminador da ProGAN simplificada
class Discriminator(nn.Module):
    # Construtor padrão
    def __init__(self):
        # Inicializa a superclasse
        super().__init__()

        # Dicionário de camadas fromRGB por resolução (projetam 1 canal -> CHAN[res])
        self.fromRGB = nn.ModuleDict()
        # Dicionário de blocos de downsample (res -> res/2)
        self.blocks = nn.ModuleDict()

        # Constroi os blocos para cada resolução
        for i, res in enumerate(RES_LIST):
            # Camada de entrada para a resolução atual (1 -> CHAN[res])
            self.fromRGB[str(res)] = nn.Conv2d(1, CHAN[res], kernel_size=1)
            # Para resoluções acima de 4x4, adiciona o bloco que reduz pela metade
            if res > 4:
                prev = RES_LIST[i-1]
                self.blocks[str(res)] = nn.Sequential(
                    # Convolução para refinar features na resolução atual
                    nn.Conv2d(CHAN[res], CHAN[res], kernel_size=3, padding=1),
                    # Ativação LeakyReLU
                    nn.LeakyReLU(0.2, inplace=True),
                    # Convolução com stride=2 para downsample para a resolução anterior
                    nn.Conv2d(CHAN[res], CHAN[prev], kernel_size=4, stride=2, padding=1),  # down x2
                    # Nova ativação
                    nn.LeakyReLU(0.2, inplace=True),
                )

        # Cabeça final que opera em 4x4 até produzir o logit
        self.final_4x4 = nn.Sequential(
            # Convolução 3x3 em 4x4
            nn.Conv2d(CHAN[4], CHAN[4], kernel_size=3, padding=1),
            # Ativação
            nn.LeakyReLU(0.2, inplace=True),
            # Achata o mapa 4x4 para um vetor
            nn.Flatten(),
            # Camada linear final que produz um logit (real/falso)
            nn.Linear(CHAN[4]*4*4, 1)  # logits
        )

    # Define o fluxo forward do discriminador
    def forward(self, x, stage: int):
        """
        x: [B, 1, H, W] com H=W=RES_LIST[stage]
        """
        # Resolução atual
        res_now = RES_LIST[stage]
        # Projeta a imagem de 1 canal para CHAN[res_now]
        h = self.fromRGB[str(res_now)](x)
        # Desce progressivamente até 4x4
        for i in range(stage, 0, -1):
            res = RES_LIST[i]       # resolução atual
            h = self.blocks[str(res)](h)  # reduz para a resolução anterior
        # Aplica a cabeça final e retorna logits [B,1]
        return self.final_4x4(h)  # [B, 1] logits


# Gera um batch de "dados reais" sintéticos para uma dada resolução
def get_real_data(resolution: int, batch_size: int):
    # Amostras 1-canal ~ Normal(0,1), apenas para exercitar o pipeline de treino
    x = torch.tensor(
        np.random.normal(0, 1, (batch_size, 1, resolution, resolution)),
        dtype=torch.float32,
        device=device
    )
    # Retorna um tensor [B,1,H,W]
    return x


# Loop de treinamento por estágios (crescimento progressivo)
def train_progan():
    # Instancia o gerador com a dimensão latente especificada
    G = Generator(latent_dim).to(device)
    # Instancia o discriminador
    D = Discriminator().to(device)

    # Função de perda binária com logits (dispensa Sigmoid nas saídas)
    criterion = nn.BCEWithLogitsLoss()
    # Otimizador para o gerador (betas típicos de GANs)
    g_opt = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    # Otimizador para o discriminador
    d_opt = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Define/seleciona o experimento no MLflow
    mlflow.set_experiment("ProGAN Training (simplified)")
    # Inicia um run de tracking
    with mlflow.start_run():
        # Itera sobre os estágios/progressões de resolução
        for stage, res in enumerate(RES_LIST):
            # Laço de épocas para o estágio atual
            for epoch in range(epochs_per_stage):
                # === Passo do Discriminador ===
                # Lê um batch de imagens reais sintéticas na resolução corrente
                real = get_real_data(res, batch_size)
                # Amostra um batch de ruído para o gerador
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                # Gera imagens falsas para este estágio (detach para não propagar gradiente para G)
                fake = G(z, stage).detach()

                # Zera gradientes do D
                d_opt.zero_grad()
                # Logits do discriminador para reais
                d_real = D(real, stage)
                # Logits do discriminador para falsos
                d_fake = D(fake, stage)
                # Perda do D com rótulos 1 para reais
                d_real_loss = criterion(d_real, torch.ones(batch_size, 1, device=device))
                # Perda do D com rótulos 0 para falsos
                d_fake_loss = criterion(d_fake, torch.zeros(batch_size, 1, device=device))
                # Perda total do D
                d_loss = d_real_loss + d_fake_loss
                # Backprop e atualização do D
                d_loss.backward()
                d_opt.step()

                # === Passo do Gerador ===
                # Novo ruído para gerar outro batch
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                # Gera imagens falsas (sem detach, agora queremos gradiente em G)
                gen = G(z, stage)
                # Zera gradientes do G
                g_opt.zero_grad()
                # Passa imagens geradas pelo D
                d_gen = D(gen, stage)
                # Objetivo do G: enganar o D (rótulos 1)
                g_loss = criterion(d_gen, torch.ones(batch_size, 1, device=device))
                # Backprop e atualização do G
                g_loss.backward()
                g_opt.step()

                # Log/print periódico a cada 10 épocas
                if epoch % 10 == 0:
                    print(f"[Stage {stage} | {res}x{res}] Epoch {epoch:03d} "
                          f"| D: {d_loss.item():.3f} (R {d_real_loss.item():.3f} F {d_fake_loss.item():.3f}) "
                          f"| G: {g_loss.item():.3f}")
                    # Registra métricas no MLflow com o sufixo da resolução
                    mlflow.log_metric(f"d_loss_{res}", d_loss.item(), step=epoch)
                    mlflow.log_metric(f"g_loss_{res}", g_loss.item(), step=epoch)

            # Loga a resolução concluída como parâmetro
            mlflow.log_param(f"resolution_stage_{stage}", res)

        # Ao final de todos os estágios, salva os modelos no MLflow
        mlflow.pytorch.log_model(G, "generator")
        mlflow.pytorch.log_model(D, "discriminator")

# Ponto de entrada do script: executa o treinamento quando chamado diretamente
if __name__ == "__main__":
    train_progan()
