# Importa o PyTorch principal.
import torch
# Importa o submódulo de camadas/arquiteturas neurais.
import torch.nn as nn
# Importa otimizadores (Adam, SGD, etc.).
import torch.optim as optim
# Importa DataLoader e um dataset simples baseado em tensores.
from torch.utils.data import DataLoader, TensorDataset
# Importa MLflow para rastrear métricas e parâmetros.
import mlflow
# Integra MLflow com PyTorch para logar/salvar o modelo.
import mlflow.pytorch

# Seleciona CUDA caso disponível; caso contrário, usa CPU.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Número de épocas de treinamento.
num_epochs = 10
# Tamanho do batch durante o treinamento.
batch_size = 64
# Taxa de aprendizado para o otimizador.
learning_rate = 0.001
# Quantidade de amostras artificiais para gerar (treino).
num_samples = 10000  # Number of artificial samples to generate
# Tamanho das imagens (224x224), compatível com a VGG.
image_size = 224  # Size of the images (224x224 pixels, matching VGG's input)
# Número de classes a prever.
num_classes = 10  # Number of classes (e.g., 10 classes)

# Função geradora de dados artificiais (imagens e rótulos).
def generate_artificial_data(num_samples, image_size, num_classes):
    # Gera imagens aleatórias (N, 3, H, W) simulando RGB.
    images = torch.randn(num_samples, 3, image_size, image_size)  # 3 channels for RGB images
    
    # Gera rótulos inteiros aleatórios no intervalo [0, num_classes-1].
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Retorna o par (imagens, rótulos).
    return images, labels

# Cria dados artificiais de treino (N amostras).
train_images, train_labels = generate_artificial_data(num_samples, image_size, num_classes)
# Cria dados artificiais de teste (N/10 amostras).
test_images, test_labels = generate_artificial_data(num_samples // 10, image_size, num_classes)

# Empacota tensores em datasets compatíveis com DataLoader.
train_dataset = TensorDataset(train_images, train_labels)
# Empacota tensores de teste em dataset.
test_dataset = TensorDataset(test_images, test_labels)
# DataLoader de treino com embaralhamento.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# DataLoader de teste sem embaralhar (ordem estável).
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a arquitetura VGG-16 (convoluções + MLP).
class VGG16(nn.Module):
    # Construtor que recebe nº de classes da saída.
    def __init__(self, num_classes=10):
        # Inicializa nn.Module.
        super().__init__()
        # Pilha convolucional de extração de características.
        self.features = nn.Sequential(
            # Bloco 1: 3→64, conv 3x3 com padding 1.
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # ReLU in-place (economiza memória).
            nn.ReLU(inplace=True),
            # Segunda conv 3x3 64→64.
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # MaxPool 2x2 (reduz metade da resolução).
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 2: 64→128, conv 3x3.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # Segunda conv 3x3 128→128.
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # MaxPool 2x2.
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 3: 128→256, conv 3x3.
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # Segunda conv 3x3 256→256.
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # Terceira conv 3x3 256→256.
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # MaxPool 2x2.
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 4: 256→512, conv 3x3.
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # Segunda conv 3x3 512→512.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # Terceira conv 3x3 512→512.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # MaxPool 2x2.
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 5: 512→512, conv 3x3.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # Segunda conv 3x3 512→512.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # Terceira conv 3x3 512→512.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # ReLU.
            nn.ReLU(inplace=True),
            # MaxPool 2x2 (com 224px de entrada, chega em 7x7 aqui).
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Classificador totalmente conectado (flatten → logits).
        self.classifier = nn.Sequential(
            # Projeção de 512×7×7 para 4096 neurônios.
            nn.Linear(512 * 7 * 7, 4096),
            # ReLU.
            nn.ReLU(inplace=True),
            # Dropout para regularização.
            nn.Dropout(),
            # Camada intermediária 4096→4096.
            nn.Linear(4096, 4096),
            # ReLU.
            nn.ReLU(inplace=True),
            # Dropout novamente.
            nn.Dropout(),
            # Camada final para num_classes logits.
            nn.Linear(4096, num_classes),
        )

    # Define o fluxo direto do modelo.
    def forward(self, x):
        # Extrai mapas de características com convoluções/pooling.
        x = self.features(x)
        # Achata para (batch, 512*7*7) antes do MLP.
        x = x.view(x.size(0), 512 * 7 * 7)
        # Classificador (camadas lineares + ReLU/Dropout).
        x = self.classifier(x)
        # Retorna logits (sem softmax; adequado a CrossEntropyLoss).
        return x

# Função que treina o modelo e loga no MLflow.
def train_model():
    # Instancia VGG16 e envia ao device.
    model = VGG16(num_classes=num_classes).to(device)
    # Função de perda para classificação multiclasse.
    criterion = nn.CrossEntropyLoss()
    # Otimizador Adam com a LR definida.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Cria/seleciona experimento no MLflow.
    mlflow.set_experiment("VGG-16 Artificial Data Classification")
    # Inicia um run (execução) do MLflow.
    with mlflow.start_run():
        # Loop de épocas.
        for epoch in range(num_epochs):
            # Modo de treino (ativa dropout/bn adequadamente).
            model.train()
            # Acumulador de loss da época.
            running_loss = 0.0
            # Contadores de acerto/total para acurácia.
            correct = 0
            total = 0
            
            # Itera sobre mini-batches do DataLoader de treino.
            for i, (images, labels) in enumerate(train_loader):
                # Move dados para o device.
                images, labels = images.to(device), labels.to(device)

                # Forward: calcula logits.
                outputs = model(images)
                # Calcula a perda do lote.
                loss = criterion(outputs, labels)

                # Zera gradientes, faz backprop e atualiza pesos.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calcula predições top-1 e acumula acertos/total.
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Acumula a perda para média da época.
                running_loss += loss.item()
                
                # Loga a cada 100 lotes: progresso, loss e acurácia acumuladas.
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")
                    mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)
                    mlflow.log_metric("train_accuracy", 100 * correct / total, step=epoch * len(train_loader) + i)

        # Loga hiperparâmetros relevantes do experimento.
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Salva o artefato do modelo treinado no MLflow.
        mlflow.pytorch.log_model(model, "vgg16_artificial_data_model")

        # Avalia no conjunto de teste (logando a métrica).
        evaluate_model(model)

# Rotina de avaliação em teste com logging de acurácia.
def evaluate_model(model):
    # Modo de avaliação (desativa dropout, usa BN em modo eval).
    model.eval()
    # Contadores de acerto e total.
    correct = 0
    total = 0
    # Sem gradiente para acelerar e economizar memória.
    with torch.no_grad():
        # Percorre o DataLoader de teste.
        for images, labels in test_loader:
            # Move dados para o device.
            images, labels = images.to(device), labels.to(device)
            # Forward: obtém logits.
            outputs = model(images)
            # Converte logits para rótulos previstos via argmax.
            _, predicted = torch.max(outputs.data, 1)
            # Atualiza contagem total e acertos.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcula a acurácia percentual final.
    accuracy = 100 * correct / total
    # Imprime a acurácia no conjunto de teste.
    print(f"Test Accuracy: {accuracy:.2f}%")
    # Loga a métrica de teste no MLflow.
    mlflow.log_metric("test_accuracy", accuracy)

# Executa treinamento e avaliação quando o script é chamado.
train_model()
