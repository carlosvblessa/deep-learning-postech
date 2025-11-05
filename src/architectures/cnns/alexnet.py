# Importa bibliotecas padrão do sistema operacional.
import os
# Importa o PyTorch (núcleo).
import torch
# Importa módulos de redes neurais e camadas.
import torch.nn as nn
# Importa otimizadores (Adam, SGD, etc.).
import torch.optim as optim
# Importa utilitários de DataLoader e dataset tensorial.
from torch.utils.data import DataLoader, TensorDataset
# Importa MLflow para rastrear experimento/execução.
import mlflow
# Importa integração do MLflow com PyTorch (para salvar o modelo).
import mlflow.pytorch

# Define o dispositivo de execução (aqui força CPU).
device = torch.device("cpu")
# Ajusta número de threads do backend para usar todos os núcleos disponíveis.
torch.set_num_threads(os.cpu_count())

# Número de épocas de treino.
num_epochs = 10
# Tamanho do lote (batch).
batch_size = 64
# Taxa de aprendizado do otimizador.
learning_rate = 0.001
# Quantidade de amostras artificiais para gerar (treino).
num_samples = 10000  # Number of artificial samples to generate
# Tamanho (altura=largura) das imagens artificiais (compatível com AlexNet).
image_size = 224  # Size of the images (224x224 pixels, matching AlexNet's input)
# Número de classes de saída.
num_classes = 10  # Number of classes (e.g., 10 classes)

# Função que cria dados artificiais (imagens e rótulos inteiros).
def generate_artificial_data(num_samples, image_size, num_classes):
    # Gera tensores com ruído gaussiano no formato [N, C, H, W] para imagens RGB.
    images = torch.randn(num_samples, 3, image_size, image_size)  # 3 channels for RGB images
    
    # Gera rótulos inteiros no intervalo [0, num_classes-1].
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Retorna imagens e rótulos.
    return images, labels

# Gera conjunto de treino artificial (N amostras).
train_images, train_labels = generate_artificial_data(num_samples, image_size, num_classes)
# Gera conjunto de teste artificial (10% do treino).
test_images, test_labels = generate_artificial_data(num_samples // 10, image_size, num_classes)

# Empacota tensores de treino em um TensorDataset (pareia imagem e rótulo).
train_dataset = TensorDataset(train_images, train_labels)
# Empacota tensores de teste em um TensorDataset.
test_dataset = TensorDataset(test_images, test_labels)
# Cria DataLoader de treino (embaralha os lotes).
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# Cria DataLoader de teste (ordem determinística).
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a arquitetura AlexNet (adaptada para PyTorch moderno).
class AlexNet(nn.Module):
    # Construtor que recebe o nº de classes.
    def __init__(self, num_classes=10):
        # Inicializa a superclasse nn.Module.
        super().__init__()
        # Bloco de extração de características (camadas convolucionais e pools).
        self.features = nn.Sequential(
            # Conv grande (11x11) com stride 4 e padding 2, 3→64 canais.
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # ReLU in-place para não linearidade.
            nn.ReLU(inplace=True),
            # MaxPool 3x3 com stride 2 para reduzir resolução.
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Segunda conv 5x5, 64→192 canais.
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # ReLU.
            nn.ReLU(inplace=True),
            # Pool 3x3 stride 2 novamente.
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Terceira conv 3x3, 192→384.
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),

            # Quarta conv 3x3, 384→256.
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),

            # Quinta conv 3x3, 256→256.
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # Pool final 3x3 stride 2.
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Classificador totalmente conectado (MLP).
        self.classifier = nn.Sequential(
            # Dropout para regularização.
            nn.Dropout(),
            # Projeção do mapa 256×6×6 para 4096 neurônios.
            nn.Linear(256 * 6 * 6, 4096),
            # ReLU.
            nn.ReLU(inplace=True),
            # Dropout novamente.
            nn.Dropout(),
            # Camada intermediária 4096→4096.
            nn.Linear(4096, 4096),
            # ReLU.
            nn.ReLU(inplace=True),
            # Camada de saída para num_classes logits.
            nn.Linear(4096, num_classes),
        )

    # Define o fluxo direto (forward) do modelo.
    def forward(self, x):
        # Extrai características com convoluções/pools.
        x = self.features(x)
        # Achata o tensor para [batch, 256*6*6].
        x = x.view(x.size(0), 256 * 6 * 6)
        # Aplica o classificador (camadas lineares).
        x = self.classifier(x)
        # Retorna logits.
        return x

# Função que treina o modelo e registra no MLflow.
def train_model():
    # Instancia a AlexNet e move para o device.
    model = AlexNet(num_classes=num_classes).to(device)
    # Define a função de perda (entropia cruzada para classificação).
    criterion = nn.CrossEntropyLoss()
    # Define o otimizador Adam com a taxa de aprendizado desejada.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Cria/seleciona experimento no MLflow.
    mlflow.set_experiment("AlexNet Artificial Data Classification")
    # Inicia um run (sessão) do MLflow.
    with mlflow.start_run():
        # Laço de épocas.
        for epoch in range(num_epochs):
            # Coloca o modelo em modo de treino.
            model.train()
            # Acumulador de perda por época.
            running_loss = 0.0
            # Contadores para acurácia.
            correct = 0
            total = 0
            
            # Itera sobre lotes do DataLoader de treino.
            for i, (images, labels) in enumerate(train_loader):
                # Move dados para o device.
                images, labels = images.to(device), labels.to(device)

                # Passagem direta: obtém logits.
                outputs = model(images)
                # Calcula a loss do lote.
                loss = criterion(outputs, labels)

                # Zera gradientes acumulados.
                optimizer.zero_grad()
                # Retropropaga o erro.
                loss.backward()
                # Atualiza os pesos.
                optimizer.step()

                # Calcula acertos no lote.
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Atualiza perda acumulada (média até o passo i).
                running_loss += loss.item()
                
                # Faz logging a cada 100 lotes (ou no lote 0).
                if i % 100 == 0:
                    # Imprime progresso (época, passo, loss e acurácia instantânea).
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")
                    # Loga a loss média até aqui (passo) no MLflow com step global.
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)
                    # Loga a acurácia acumulada até aqui (passo) no MLflow.
                    mlflow.log_metric("train_accuracy", 100 * correct / total, step=epoch * len(train_loader) + i)

        # Loga hiperparâmetros importantes do experimento.
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Salva o modelo treinado no artefato do MLflow.
        mlflow.pytorch.log_model(model, "alexnet_artificial_data_model")

        # Avalia o modelo no conjunto de teste (ainda dentro do run).
        evaluate_model(model)

# Função de avaliação no conjunto de teste com logging da acurácia.
def evaluate_model(model):
    # Coloca o modelo em modo de avaliação (desativa dropout).
    model.eval()
    # Zera contadores de acerto/total.
    correct = 0
    total = 0
    # Desliga gradientes para acelerar/otimizar.
    with torch.no_grad():
        # Itera sobre lotes de teste.
        for images, labels in test_loader:
            # Move dados para o device.
            images, labels = images.to(device), labels.to(device)
            # Forward para obter logits.
            outputs = model(images)
            # Converte logits em rótulos previstos (argmax).
            _, predicted = torch.max(outputs.data, 1)
            # Atualiza contagem de amostras.
            total += labels.size(0)
            # Soma acertos do lote.
            correct += (predicted == labels).sum().item()

    # Calcula acurácia percentual.
    accuracy = 100 * correct / total
    # Exibe no console a acurácia de teste.
    print(f"Test Accuracy: {accuracy:.2f}%")
    # Loga a acurácia de teste no MLflow (usa run ativo do chamador).
    mlflow.log_metric("test_accuracy", accuracy)

# Ponto de entrada do script (executa treino quando chamado diretamente).
if __name__ == "__main__":
    # Dispara o processo de treinamento/com logging e avaliação.
    train_model()
