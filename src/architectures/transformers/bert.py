# Caminho do arquivo (informativo)
# src/architectures/transformers/bert.py

# Importa o núcleo do PyTorch (tensores e utilidades)
import torch
# Importa módulos de camadas/containers de rede
import torch.nn as nn
# Importa o otimizador Adam
from torch.optim import Adam
# Importa DataLoader/TensorDataset para pipeline de dados
from torch.utils.data import DataLoader, TensorDataset
# Importa BERT base, tokenizer e config do Hugging Face
from transformers import BertModel, BertTokenizer, BertConfig
# Importa NumPy para utilidades com arrays e sementes
import numpy as np
# MLflow para experiment tracking
import mlflow
# Submódulo do MLflow para serializar modelos PyTorch
import mlflow.pytorch
# Importa gerador pseudoaleatório padrão do Python (seed global)
import random

# Define o dispositivo de execução como string ("cpu" ou "cuda")
device = "cpu"
# Define a semente global para reprodutibilidade
SEED = 42
# Fixa a semente do gerador do Python
random.seed(SEED)
# Fixa a semente do NumPy
np.random.seed(SEED)
# Fixa a semente do PyTorch (CPU)
torch.manual_seed(SEED)
# Se o device for CUDA, fixa as sementes das GPUs também
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Nome do checkpoint do BERT base uncased
model_name = "bert-base-uncased"
# Número de classes para classificação
num_classes = 2
# Número de épocas de treinamento
num_epochs = 5
# Tamanho do minibatch
batch_size = 32
# Taxa de aprendizado do otimizador
learning_rate = 1e-4
# Comprimento máximo das sequências de entrada
sequence_length = 20
# Quantidade de amostras sintéticas a gerar
num_samples = 1000
# Tamanho da camada oculta da cabeça MLP
hidden_size = 50  # tamanho da camada oculta da cabeça MLP

# Cria o tokenizer do BERT a partir do checkpoint
tokenizer = BertTokenizer.from_pretrained(model_name)

# Carrega a configuração do BERT
config = BertConfig.from_pretrained(model_name)
# Carrega o BERT base e move para o device
bert_model = BertModel.from_pretrained(model_name, config=config).to(device)

# Congela todos os parâmetros do BERT (somente a cabeça será treinada)
for p in bert_model.parameters():
    p.requires_grad = False
# Coloca o BERT em modo de avaliação (economiza memória/tempo)
bert_model.eval()

# Obtém o tamanho do hidden do BERT (geralmente 768)
bert_hidden = bert_model.config.hidden_size  # geralmente 768

# Define um gerador de dados artificiais para classificação
def generate_artificial_data(num_samples, sequence_length, num_classes):
    # Cria frases simples repetindo tokens para exercitar o pipeline
    sentences = [" ".join([f"word_{j}"] * sequence_length) for j in range(num_samples)]
    # Amostra rótulos inteiros no intervalo [0, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))

    # Tokeniza as sentenças com padding/truncation até sequence_length
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    # Extrai ids de tokens
    input_ids = enc["input_ids"]
    # Extrai máscara de atenção (1 = token válido, 0 = padding)
    attention_mask = enc["attention_mask"]
    # Retorna tensores prontos para DataLoader
    return input_ids, attention_mask, labels

# Gera conjunto de treino sintético
train_input_ids, train_attention_mask, train_labels = generate_artificial_data(
    num_samples, sequence_length, num_classes
)
# Gera conjunto de teste/validação sintético (10% do treino)
test_input_ids, test_attention_mask, test_labels = generate_artificial_data(
    num_samples // 10, sequence_length, num_classes
)

# Cria o TensorDataset de treino (inputs, máscara, labels)
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
# Cria o TensorDataset de teste (inputs, máscara, labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

# Habilita pin_memory apenas quando o device for CUDA
pin_mem = device == "cuda"
# DataLoader de treino com embaralhamento
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
# DataLoader de teste sem embaralhamento
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

# Define a cabeça de classificação sobre o embedding [CLS] do BERT
class BERTClassifier(nn.Module):
    # Construtor recebe o BERT congelado, tamanho do hidden da cabeça e #classes
    def __init__(self, bert_model, hidden_size, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)
        # (Opcional) pequeno dropout ajuda em estabilidade
        self.dropout = nn.Dropout(p=0.1)

    # Forward: extrai [CLS] do BERT e passa pela MLP classificadora
    def forward(self, input_ids, attention_mask):
        # BERT congelado: desliga grad para economizar memória
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Seleciona o vetor do token [CLS] (posição 0)
            cls = outputs.last_hidden_state[:, 0, :]  # token [CLS]

        # Passa por FC + ReLU + Dropout e camadas de classificação
        x = self.relu(self.fc(cls))
        x = self.dropout(x)
        logits = self.classifier(x)
        # Retorna logits por classe
        return logits

# Define rotina de avaliação (perda média e acurácia)
def evaluate_model(model, criterion):
    # Modo avaliação (desativa dropout, etc.)
    model.eval()
    # Acumulador de perda total
    test_loss = 0.0
    # Contador de acertos
    correct = 0
    # Contador de exemplos
    total = 0

    # Desativa gradientes para acelerar a inferência
    with torch.inference_mode():
        # Itera sobre o DataLoader de teste
        for input_ids, attention_mask, labels in test_loader:
            # Move tensores para o device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Obtém logits da cabeça de classificação
            outputs = model(input_ids, attention_mask)
            # Calcula a perda do batch
            loss = criterion(outputs, labels)
            # Acumula a perda
            test_loss += loss.item()

            # Predição é o índice do maior logit
            predicted = outputs.argmax(dim=1)
            # Atualiza total de exemplos
            total += labels.size(0)
            # Atualiza acertos
            correct += (predicted == labels).sum().item()

    # Perda média por batch
    avg_loss = test_loss / len(test_loader)
    # Acurácia em porcentagem
    acc = 100.0 * correct / total
    # Exibe métricas no console
    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%")

    # Registra métricas no MLflow
    mlflow.log_metric("test_loss", avg_loss)
    mlflow.log_metric("test_accuracy", acc)

# Define rotina principal de treinamento
def train_model():
    # Instancia o classificador com BERT congelado
    model = BERTClassifier(bert_model, hidden_size, num_classes).to(device)
    # Usa CrossEntropyLoss para classificação multiclasse
    criterion = nn.CrossEntropyLoss()

    # Otimize apenas a cabeça (fc + classifier)
    optimizer = Adam(
        list(model.fc.parameters()) + list(model.classifier.parameters()),
        lr=learning_rate,
    )

    # Define/seleciona o experimento no MLflow
    mlflow.set_experiment("BERT Artificial Data Classification")
    # Abre um run para registrar parâmetros, métricas e artefatos
    with mlflow.start_run():
        # Log de hiperparâmetros e configs
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("bert_hidden", bert_hidden)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("freeze_bert", True)

        # Loop de épocas
        for epoch in range(num_epochs):
            # Modo treino (ativa dropout da cabeça)
            model.train()
            # Acumuladores de perda e acurácia
            running_loss = 0.0
            correct = 0
            total = 0

            # Itera sobre os batches do DataLoader de treino
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                # Move tensores para o device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                # Forward pela cabeça de classificação
                outputs = model(input_ids, attention_mask)
                # Calcula perda do batch
                loss = criterion(outputs, labels)

                # Zera gradientes
                optimizer.zero_grad()
                # Backprop da perda
                loss.backward()
                # Atualiza pesos da cabeça
                optimizer.step()

                # Acumula perda
                running_loss += loss.item()
                # Atualiza métricas online
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Logging ocasional por batch
                if i % 100 == 0:
                    batch_acc = 100.0 * correct / total
                    # Exibe progresso no console
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}] "
                        f"Step [{i+1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%"
                    )
                    # Loga perda/acurácia do batch no MLflow
                    mlflow.log_metric("train_batch_loss", loss.item())
                    mlflow.log_metric("train_batch_accuracy", batch_acc)

            # Calcula métricas médias da época
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100.0 * correct / total
            # Exibe resumo da época
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"- Avg Loss: {epoch_loss:.4f} | Avg Acc: {epoch_acc:.2f}%"
            )
            # Loga métricas por época no MLflow
            mlflow.log_metric("train_epoch_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_epoch_accuracy", epoch_acc, step=epoch)

        # Salva o modelo completo (BERT congelado + cabeça) como artefato
        mlflow.pytorch.log_model(model, "bert_artificial_data_model")

        # Avaliação final no conjunto de teste
        evaluate_model(model, criterion)

# Ponto de entrada: executa o treinamento quando chamado como script
if __name__ == "__main__":
    train_model()
