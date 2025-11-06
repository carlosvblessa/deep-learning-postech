# Caminho do arquivo (informativo)
# src/architectures/transformers/gpt.py

# Importa funções matemáticas (p.ex., exp para perplexidade)
import math
# Importa o gerador pseudoaleatório padrão do Python
import random
# Importa utilitários do sistema operacional
import os

# Importa NumPy para utilidades com arrays e seeds
import numpy as np
# Importa o núcleo do PyTorch (tensores e kernel numérico)
import torch
# Importa módulos de camadas/containers de rede
import torch.nn as nn
# Importa o otimizador Adam para atualização de pesos
from torch.optim import Adam
# Importa DataLoader/TensorDataset para pipeline de dados
from torch.utils.data import DataLoader, TensorDataset
# Importa GPT-2 (LM causal) e seu tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# MLflow para experiment tracking
import mlflow
# Submódulo do MLflow para serializar modelos PyTorch
import mlflow.pytorch

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
# Se o device for CUDA, fixa as sementes da(s) GPU(s)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)
    
# Nome do checkpoint base do modelo (GPT-2 pequeno)
model_name = "gpt2"
# Número de épocas de treinamento
num_epochs = 5
# Tamanho do minibatch
batch_size = 32
# Taxa de aprendizado do otimizador
learning_rate = 1e-4
# Comprimento máximo das sequências (tokens)
sequence_length = 20
# Quantidade de amostras sintéticas a gerar
num_samples = 1000

# Cria o tokenizer do GPT-2 a partir do checkpoint indicado
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# GPT-2 não possui pad_token por padrão; usa eos_token como pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Define o padding à esquerda (convencional para modelos causais)
tokenizer.padding_side = "left"

# Carrega o modelo GPT-2 com cabeça de LM (causal) e move para o device
gpt = GPT2LMHeadModel.from_pretrained(model_name).to(device)
# Informa ao modelo qual é o pad_token_id (impacta máscaras internas)
gpt.config.pad_token_id = tokenizer.pad_token_id

# Define um gerador de dados artificiais para exercitar o pipeline de LM
def generate_artificial_data(num_samples: int, sequence_length: int):
    """
    Gera um conjunto de sentenças artificiais para exercitar o pipeline de LM.
    Labels = input_ids, com padding mascarado por -100 (ignore_index).
    """
    # Cria frases simples com leve variação de tokens
    sentences = [
        " ".join([f"word_{j % 10}"] * max(1, sequence_length // 5))
        for j in range(num_samples)
    ]

    # Tokeniza o lote de sentenças com padding/truncation para sequence_length
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )

    # Extrai os ids de tokens
    input_ids = enc["input_ids"]
    # Extrai a máscara de atenção (1 = token válido, 0 = padding)
    attention_mask = enc["attention_mask"]

    # Define labels iguais a input_ids (LM causal), mascarando o padding com -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # ignore_index da CrossEntropy

    # Retorna tensores prontos para DataLoader
    return input_ids, attention_mask, labels

# Gera o conjunto sintético de treino
train_input_ids, train_attention_mask, train_labels = generate_artificial_data(
    num_samples, sequence_length
)
# Gera o conjunto sintético de teste/validação (10% do treino, no mínimo 1)
test_input_ids, test_attention_mask, test_labels = generate_artificial_data(
    max(1, num_samples // 10), sequence_length
)

# Cria o TensorDataset de treino: (inputs, mask, labels)
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
# Cria o TensorDataset de teste: (inputs, mask, labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

# Habilita pin_memory quando o device for CUDA (acelera cópias host->GPU)
pin_mem = device == "cuda"
# DataLoader de treino com embaralhamento e 2 workers
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem, num_workers=2
)
# DataLoader de teste sem embaralhamento
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem, num_workers=2
)

# Define um wrapper simples para expor (loss, logits) de forma uniforme
class GPTLanguageModel(nn.Module):
    # Construtor recebe uma instância de GPT-2 já carregada
    def __init__(self, gpt_model: GPT2LMHeadModel):
        super().__init__()
        self.gpt = gpt_model

    # Forward compatível com treino/val: retorna (loss, logits)
    def forward(self, input_ids, attention_mask, labels=None):
        # Passa pelo GPT-2; se labels fornecidos, ele calcula a loss internamente
        out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Retorna a perda (se houver) e os logits token-a-token
        return out.loss, out.logits

# Define a rotina de avaliação no conjunto de teste
def evaluate_model(model: GPTLanguageModel):
    # Coloca o modelo em modo de avaliação (desativa dropout, etc.)
    model.eval()
    # Acumulador de perda total
    test_loss = 0.0
    # Contador de minibatches
    n_batches = 0

    # Desativa gradientes para acelerar a inferência
    with torch.inference_mode():
        # Itera sobre os batches do DataLoader de teste
        for input_ids, attention_mask, labels in test_loader:
            # Move input_ids para o device (transferência não bloqueante)
            input_ids = input_ids.to(device, non_blocking=True)
            # Move attention_mask para o device
            attention_mask = attention_mask.to(device, non_blocking=True)
            # Move labels para o device
            labels = labels.to(device, non_blocking=True)

            # Calcula a loss do batch; logits não são necessários aqui
            loss, _ = model(input_ids, attention_mask, labels=labels)
            # Acumula a perda
            test_loss += loss.item()
            # Incrementa o número de batches
            n_batches += 1

    # Calcula a perda média por batch
    avg_loss = test_loss / max(1, n_batches)
    # Converte para perplexidade (evita overflow para perdas muito altas)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")  # evita overflow
    # Exibe métricas de avaliação
    print(f"Test Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")
    # Registra métricas no MLflow
    mlflow.log_metric("test_loss", avg_loss)
    mlflow.log_metric("test_perplexity", ppl)

# Define a rotina principal de treinamento
def train_model():
    # Envolve o GPT-2 no wrapper para interface (loss, logits)
    model = GPTLanguageModel(gpt).to(device)
    # Cria o otimizador Adam com a taxa de aprendizado especificada
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Define/seleciona o experimento no MLflow
    mlflow.set_experiment("GPT Artificial Data Language Modeling")
    # Abre um run para registrar parâmetros, métricas e artefatos
    with mlflow.start_run():
        # Registra hiperparâmetros e configs relevantes
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("padding_side", tokenizer.padding_side)
        mlflow.log_param("pad_token", tokenizer.pad_token)

        # Contador global de passos (para métricas por batch)
        global_step = 0
        # Loop de épocas
        for epoch in range(num_epochs):
            # Modo treino
            model.train()
            # Acumulador de perda por época
            running_loss = 0.0

            # Itera sobre batches do DataLoader de treino
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                # Move tensores para o device
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Forward + perda supervisionada (LM causal)
                loss, _ = model(input_ids, attention_mask, labels=labels)

                # Zera gradientes de forma eficiente
                optimizer.zero_grad(set_to_none=True)
                # Backpropaga a perda
                loss.backward()
                # Clipping de gradiente para estabilidade numérica
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Atualiza os pesos
                optimizer.step()

                # Acumula perda deste batch
                running_loss += loss.item()

                # Logging ocasional a cada 100 passos
                if i % 100 == 0:
                    # Exibe progresso no console
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}] "
                        f"Step [{i+1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )
                    # Loga a perda do batch no MLflow (step = global_step)
                    mlflow.log_metric("train_batch_loss", loss.item(), step=global_step)

                # Incrementa o step global
                global_step += 1

            # Calcula a perda média por época
            epoch_avg_loss = running_loss / len(train_loader)
            # Exibe a perda média no console
            print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {epoch_avg_loss:.4f}")
            # Loga a perda média da época (step = epoch)
            mlflow.log_metric("train_epoch_loss", epoch_avg_loss, step=epoch)

        # Salva o GPT-2 (fine-tuned, se houver) como artefato do MLflow
        mlflow.pytorch.log_model(model.gpt, "fine_tuned_gpt_model")

        # Executa avaliação final no conjunto de teste
        evaluate_model(model)

# Ponto de entrada: executa o treinamento quando chamado como script
if __name__ == "__main__":
    train_model()
