# Importa o PyTorch principal.
import torch
# Importa os módulos de redes neurais (camadas, etc.).
import torch.nn as nn
# Importa os otimizadores (Adam, SGD, etc.).
import torch.optim as optim
# Importa utilitários de DataLoader e dataset baseados em tensores.
from torch.utils.data import DataLoader, TensorDataset
# Importa o MLflow para registrar métricas/parâmetros.
import mlflow
# Integração do MLflow com PyTorch (para salvar modelos).
import mlflow.pytorch

# Seleciona o dispositivo: CUDA se disponível, senão CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Número de épocas de treino.
num_epochs = 20
# Tamanho do lote (batch).
batch_size = 128
# Taxa de aprendizado do otimizador.
learning_rate = 0.05
# Quantidade de amostras artificiais a gerar para treino.
num_samples = 10000  # Number of artificial samples to generate
# Tamanho de entrada das imagens (32x32) — compatível com LeNet clássico.
image_size = 32  # Size of the images (32x32 pixels)
# Número de classes (por ex., 10 classes para dígitos 0–9).
num_classes = 10  # Number of classes (e.g., 10 classes for digits 0-9)

# Define a arquitetura LeNet (conv-conv-MLP).
class LeNet(nn.Module):
    # Construtor recebendo o nº de classes de saída.
    def __init__(self, num_classes=10):
        # Inicializa a superclasse nn.Module.
        super().__init__()
        # Primeira convolução (1 canal → 6 mapas), kernel 5x5.
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Segunda convolução (6 → 16), kernel 5x5.
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Primeira camada totalmente conectada (flatten 16×5×5 → 120).
        self.fc1 = nn.Linear(16*5*5, 120)
        # Segunda camada totalmente conectada (120 → 84).
        self.fc2 = nn.Linear(120, 84)
        # Ativação ReLU.
        self.relu = nn.ReLU()
        # Camada de saída (84 → num_classes).
        self.fc3 = nn.Linear(84, num_classes)
        # MaxPooling 2×2 com stride 2.
        self.pool = nn.MaxPool2d(2, 2)
        # ⚠️ LogSoftmax (log-probabilidades). Se usar CrossEntropyLoss adiante, o ideal é NÃO aplicar LogSoftmax aqui (ou trocar a loss para NLLLoss).
        self.softmax = nn.LogSoftmax(dim=1)

    # Define o fluxo direto (forward).
    def forward(self, x):
        # Convolução 1 → ReLU → MaxPool.
        x = self.pool(self.relu(self.conv1(x)))
        # Convolução 2 → ReLU → MaxPool.
        x = self.pool(self.relu(self.conv2(x)))
        # Achata para vetor (batch, 16*5*5).
        x = x.view(-1, 16*5*5)
        # FC1 + ReLU.
        x = self.relu(self.fc1(x))
        # FC2 + ReLU.
        x = self.relu(self.fc2(x))
        # Camada de saída (logits).
        # x = self.fc3(x)
        return self.fc3(x)  # logits puros
        # ⚠️ Retorna log-probabilidades; compatível com NLLLoss, mas NÃO com CrossEntropyLoss (que já inclui log-softmax internamente).
        # return self.softmax(x)



# Gera dados artificiais (imagens aleatórias e rótulos inteiros).
def generate_artificial_data(num_samples, image_size, num_classes):
    # Imagens aleatórias com 1 canal, shape [N, 1, H, W].
    images = torch.randn(num_samples, 1, image_size, image_size)
    # Rótulos inteiros no intervalo [0, num_classes-1].
    labels = torch.randint(0, num_classes, (num_samples,))
    # Retorna tensores de imagens e rótulos.
    return images, labels

# Cria o conjunto de treino artificial.
train_images, train_labels = generate_artificial_data(
    num_samples,
    image_size,
    num_classes
)

# Cria o conjunto de teste artificial (10% do treino).
test_images, test_labels = generate_artificial_data(
    num_samples // 10,
    image_size,
    num_classes
)

# Empacota treino em TensorDataset (pareia imagem↔rótulo).
train_dataset = TensorDataset(train_images, train_labels)
# Empacota teste em TensorDataset.
test_dataset = TensorDataset(test_images, test_labels)
# DataLoader de treino com embaralhamento.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# DataLoader de teste sem embaralhar.
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Função de treinamento principal.
def train_model():
    # Instancia o modelo e move para o dispositivo.
    model = LeNet().to(device)
    # Define a função de perda.
    criterion = nn.CrossEntropyLoss()
    # Define o otimizador (Adam) com a LR escolhida.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define/seleciona experimento no MLflow.
    mlflow.set_experiment("LeNet Artificial Data Classification")
    # Inicia um run no MLflow para agrupar métricas/artefatos.
    with mlflow.start_run():
        # Loop por épocas.
        for epoch in range(num_epochs):
            # Coloca o modelo em modo de treino.
            model.train()
            # Acumulador de loss da época.
            running_loss = 0.0
            # Contadores para acurácia.
            correct = 0
            total = 0
            
            # Itera sobre os lotes do DataLoader.
            for i, (images, labels) in enumerate(train_loader):
                # Move dados para o device (CPU/GPU).
                images, labels = images.to(device), labels.to(device)

                # Forward: obtém predições.
                outputs = model(images)
                # Calcula a perda do lote.
                loss = criterion(outputs, labels)

                # Zera gradientes acumulados.
                optimizer.zero_grad()
                # Backprop: calcula gradientes.
                loss.backward()
                # Atualiza os pesos.
                optimizer.step()

                # Converte logits/log-probs em classes previstas.
                _, predicted = torch.max(outputs.data, 1)
                # Atualiza total de exemplos.
                total += labels.size(0)
                # Soma acertos do lote.
                correct += (predicted == labels).sum().item()

                # Acumula a perda.
                running_loss += loss.item()
                
                # Faz logging a cada 100 lotes.
                if i % 100 == 0:
                    # Imprime progresso (época, passo, loss e acurácia acumulada).
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")
                    # Loga perda média até aqui (passo global) no MLflow.
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)
                    # Loga acurácia acumulada até aqui.
                    mlflow.log_metric("train_accuracy", 100 * correct / total, step=epoch * len(train_loader) + i)

        # Registra hiperparâmetros usados no experimento.
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Salva o modelo como artefato do MLflow.
        mlflow.pytorch.log_model(model, "lenet_artificial_data_model")

        # Avalia no conjunto de teste (mantém o run aberto para logar métrica).
        evaluate_model(model)

# Função de avaliação no conjunto de teste.
def evaluate_model(model):
    # Modo de avaliação (desativa dropout, etc.).
    model.eval()
    # Zera contadores.
    correct = 0
    total = 0
    # Desliga gradiente para acelerar e economizar memória.
    with torch.no_grad():
        # Itera pelos lotes do conjunto de teste.
        for images, labels in test_loader:
            # Move dados para o device.
            images, labels = images.to(device), labels.to(device)
            # Forward: obtém saídas do modelo.
            outputs = model(images)
            # Predição de classe por argmax.
            _, predicted = torch.max(outputs.data, 1)
            # Atualiza totais.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcula a acurácia percentual final.
    accuracy = 100 * correct / total
    # Exibe a acurácia no teste.
    print(f"Test Accuracy: {accuracy:.2f}%")
    # Loga a métrica de teste no MLflow (no run ativo).
    mlflow.log_metric("test_accuracy", accuracy)

# Executa o treinamento e a avaliação quando o script é chamado diretamente.
train_model()
