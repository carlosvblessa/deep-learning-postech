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
# Fixa a semente para reprodutibilidade básica
torch.manual_seed(42)
np.random.seed(42)

# Parâmetros dos dados (distribuição alvo que o G deve imitar)
# Usaremos amostras gaussianas 1D por feature com média 4.0 e desvio 1.25.
data_mean = 4.0
data_stddev = 1.25

# Hiperparâmetros dos modelos e do treino
# Dimensão do ruído que entra no Gerador
g_input_size   = 50
# Capacidade (largura) do Gerador
g_hidden_size  = 100
# Dimensão de saída do Gerador (precisa casar com a entrada do Discriminador)
g_output_size  = 50
# Dimensão de entrada do Discriminador (tem que ser igual a g_output_size)
d_input_size   = 50
# Camadas ocultas do Discriminador
d_hidden_size  = 50
d_hidden_2_size = 50
# Saída do Discriminador (logit único)
d_output_size  = 1
# Tamanho do minibatch, nº de épocas e LR do Adam
minibatch_size = 50
num_epochs     = 1000
lr             = 2e-4

# Função que gera dados "reais" do domínio (alvo do treinamento)
# Retorna um batch [B, d_input_size] de amostras gaussianas.
def get_real_data():
    x = torch.tensor(
        np.random.normal(data_mean, data_stddev, (minibatch_size, d_input_size)),
        dtype=torch.float32,
        device=device
    )
    return x

# Definição do Gerador (MLP simples)
# Entrada: ruído z ∈ R^{g_input_size}
# Saída: vetor "falso" ∈ R^{g_output_size}
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_input_size, g_hidden_size),
            nn.ReLU(True),
            nn.Linear(g_hidden_size, g_hidden_size),
            nn.ReLU(True),
            nn.Linear(g_hidden_size, g_output_size)  # logits contínuos (sem ativação final)
        )
    def forward(self, z):
        return self.net(z)

# Definição do Discriminador (MLP)
# Entrada: amostra x ∈ R^{d_input_size}
# Saída: logit escalar (probabilidade de "real" após Sigmoid implícito na loss)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input_size, d_hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_hidden_size, d_hidden_2_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_hidden_2_size, 1)  # logits (usar BCEWithLogitsLoss)
        )
    def forward(self, x):
        return self.net(x)

# Laço de treinamento da GAN (não-condicional)
def train():
    # Instancia modelos no device
    G = Generator().to(device)
    D = Discriminator().to(device)

    # BCEWithLogitsLoss: combina Sigmoid + BCE de forma estável numericamente
    criterion = nn.BCEWithLogitsLoss()

    # Otimizadores independentes para D e G (correto)
    d_optimizer = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    # Cria/seleciona experimento no MLflow
    mlflow.set_experiment("GAN Training (1D)")
    with mlflow.start_run():
        # Loop principal de épocas
        for epoch in range(num_epochs):
            # -------- Passo do Discriminador (D) --------
            # Amostras reais do domínio alvo
            real_data = get_real_data()  # [B, 50]
            # Amostras falsas geradas a partir de ruído
            noise = torch.randn(minibatch_size, g_input_size, device=device)
            fake_data = G(noise).detach()  # detach: não propaga gradiente para G aqui

            # Zera gradientes de D
            d_optimizer.zero_grad()
            # Logits de D para reais e falsos
            d_real = D(real_data)
            d_fake = D(fake_data)
            # Labels para loss: reais=1, falsos=0
            d_real_loss = criterion(d_real, torch.ones(minibatch_size, 1, device=device))
            d_fake_loss = criterion(d_fake, torch.zeros(minibatch_size, 1, device=device))
            # Loss total de D
            d_loss = d_real_loss + d_fake_loss
            # Backprop e passo do otimizador de D
            d_loss.backward()
            d_optimizer.step()

            # -------- Passo do Gerador (G) --------
            # Novo ruído para evitar colapso de batch
            noise = torch.randn(minibatch_size, g_input_size, device=device)
            gen = G(noise)
            # Zera gradientes de G
            g_optimizer.zero_grad()
            # Passa os gerados por D — objetivo de G é "enganar" D (alvos=1)
            d_gen = D(gen)
            g_loss = criterion(d_gen, torch.ones(minibatch_size, 1, device=device))
            # Backprop e passo do otimizador de G
            g_loss.backward()
            g_optimizer.step()

            # Logging/print periódico
            if epoch % 100 == 0:
                print(f"Epoch {epoch:04d} | D: real {d_real_loss.item():.3f} "
                      f"fake {d_fake_loss.item():.3f} tot {d_loss.item():.3f} | "
                      f"G: {g_loss.item():.3f}")
                mlflow.log_metric("d_real_loss", d_real_loss.item(), step=epoch)
                mlflow.log_metric("d_fake_loss", d_fake_loss.item(), step=epoch)
                mlflow.log_metric("d_loss", d_loss.item(), step=epoch)
                mlflow.log_metric("g_loss", g_loss.item(), step=epoch)

        # Log de hiperparâmetros no fim do treinamento
        mlflow.log_params({
            "g_input_size": g_input_size,
            "g_hidden_size": g_hidden_size,
            "g_output_size": g_output_size,
            "d_input_size": d_input_size,
            "d_hidden_size": d_hidden_size,
            "d_hidden_2_size": d_hidden_2_size,
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
