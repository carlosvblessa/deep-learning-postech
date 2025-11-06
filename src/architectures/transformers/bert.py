# Caminho do arquivo (informativo)
# src/architectures/transformers/bert.py

# Importa o núcleo do PyTorch (tensores e kernel numérico)
import torch
# Importa módulos de camadas e containers de rede
import torch.nn as nn
# Importa o otimizador Adam para atualização de pesos
from torch.optim import Adam
# Importa DataLoader e TensorDataset para pipeline de dados
from torch.utils.data import DataLoader, TensorDataset
# Importa o modelo e o tokenizer do GPT-2 (LM causal)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Importa NumPy para operações auxiliares com arrays
import numpy as np
# MLflow para experiment tracking
import mlflow
# Submódulo do MLflow para serializar modelos PyTorch
import mlflow.pytorch
# Importa funções matemáticas (p.ex., exp para perplexidade)
import math
# Importa utilitários de sistema operacional
import os
# Importa gerador pseudoaleatório padrão do Python
import random

# Seleciona o dispositivo de execução: GPU se disponível, senão CPU
device = "cpu"
# Define a semente global para reprodutibilidade
SEED = 42
# Fixa a semente do gerador pseudoaleatório do Python
random.seed(SEED)
# Fixa a semente do NumPy
np.random.seed(SEED)
# Fixa a semente do PyTorch
torch.manual_seed(SEED)
# Se houver CUDA, fixa também as sementes de todos os dispositivos CUDA
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
# Comprimento máximo das sequências de entrada/label
sequence_length = 20
# Quantidade de amostras sintéticas a gerar para treino
num_samples = 1000

# Cria o tokenizer do GPT-2 a partir do checkpoint (vocabulário + merges)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# GPT-2 não possui pad_token por padrão; mapeia pad_token para eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Define o padding à esquerda (necessário/convencional em modelos causais)
tokenizer.padding_side = "left"

# Carrega o GPT-2 com cabeça de linguagem (LMHead) para treinamento causal
gpt = GPT2LMHeadModel.from_pretrained(model_name).to(device)
# Garante que o modelo conheça o pad_token_id, alinhado ao tokenizer
gpt.config.pad_token_id = tokenizer.pad_token_id

# Define um gerador de dados artificiais para exercitar o pipeline
def generate_artificial_data(num_samples, sequence_length):
    # Cria sentenças simples e repetitivas com pequena variação de tokens
    sentences = [
        " ".join([f"word_{j % 10}"] * max(1, sequence_length // 5))
        for j in range(num_samples)
    ]

    # Tokeniza em lote, aplicando padding/truncamento ao tamanho alvo
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )

    # Extrai input_ids (tokens) do batch tokenizado
    input_ids = enc["input_ids"]
    # Extrai atenção (1 = token real, 0 = padding)
    attention_mask = enc["attention_mask"]

    # Define labels iguais aos input_ids para LM causal (shift interno no modelo)
    labels = input_ids.clone()
    # Mascara os tokens de padding com -100 (ignore_index da CrossEntropyLoss)
    labels[attention_mask == 0] = -100

    # Retorna tensores prontos para DataLoader: inputs, máscara e labels
    return input_ids, attention_mask, labels

# Gera o conjunto de treino sintético
train_input_ids, train_attention_mask, train_labels = generate_artificial_data(
    num_samples, sequence_length
)
# Gera o conjunto de teste/validação sintético (menor)
test_input_ids, test_attention_mask, test_labels = generate_artificial_data(
    num_samples // 10, sequence_length
)

# Cria o TensorDataset de treino com (inputs, máscara, labels)
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
# Cria o TensorDataset de teste com (inputs, máscara, labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

# Ativa pin_memory no DataLoader quando estiver em GPU para acelerar cópias
pin_mem = device == "cuda"
# Cria o DataLoader de treino com embaralhamento e workers paralelos
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem, num_workers=2
)
# Cria o DataLoader de teste sem embaralhamento
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem, num_workers=2
)

# Define um wrapper simples para isolar chamadas ao GPT-2
class GPTLanguageModel(nn.Module):
    # Construtor recebe uma instância de GPT-2 já carregada
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt = gpt_model

    # Forward compatível com treino/val: retorna (loss, logits)
    def forward(self, input_ids, attention_mask, labels=None):
        # Propaga pelos blocos do GPT-2 com labels (o modelo calcula loss internamente)
        out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Retorna a perda (se labels fornecidos) e os logits token-a-token
        return out.loss, out.logits

# Define a rotina de avaliação no conjunto de teste
def evaluate_model(model):
    # Coloca o modelo em modo de avaliação (desativa dropout, etc.)
    model.eval()
    # Acumula a perda total para média posterior
    test_loss = 0.0
    # Conta o número de minibatches processados
    n_batches = 0

    # Desativa gradientes para acelerar a avaliação
    with torch.inference_mode():
        # Itera sobre os batches do DataLoader de teste
        for input_ids, attention_mask, labels in test_loader:
            # Move input_ids para o device (GPU/CPU) com transferência não bloqueante
            input_ids = input_ids.to(device, non_blocking=True)
            # Move attention_mask para o device
            attention_mask = attention_mask.to(device, non_blocking=True)
            # Move labels para o device
            labels = labels.to(device, non_blocking=True)

            # Computa a loss do batch; logits não são necessários aqui
            loss, _ = model(input_ids, attention_mask, labels=labels)
            # Soma a perda para média
            test_loss += loss.item()
            # Incrementa o contador de batches
            n_batches += 1

    # Calcula a perda média por minibatch
    avg_loss = test_loss / max(1, n_batches)
    # Converte perda média em perplexidade (protege contra overflow numérico)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    # Exibe métricas de avaliação no console
    print(f"Test Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")
    # Registra métricas no MLflow
    mlflow.log_metric("test_loss", avg_loss)
    mlflow.log_metric("test_perplexity", ppl)

# Define a rotina principal de treinamento
def train_model():
    # Envolve o GPT-2 no wrapper para interface uniforme (loss, logits)
    model = GPTLanguageModel(gpt).to(device)
    # Cria o otimizador Adam sobre todos os parâmetros treináveis
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Define/seleciona o experimento no MLflow
    mlflow.set_experiment("GPT Artificial Data Language Modeling")
    # Abre um run para agrupar parâmetros, métricas e artefatos
    with mlflow.start_run():
        # Loga hiperparâmetros e configurações relevantes
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("padding_side", tokenizer.padding_side)
        mlflow.log_param("pad_token", tokenizer.pad_token)

        # Contador global de passos (útil para métricas por batch)
        global_step = 0
        # Loop principal de épocas
        for epoch in range(num_epochs):
            # Coloca o modelo em modo de treino
            model.train()
            # Zera o acumulador de perda por época
            running_loss = 0.0

            # Itera sobre os batches do DataLoader de treino
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                # Move input_ids para o device
                input_ids = input_ids.to(device, non_blocking=True)
                # Move attention_mask para o device
                attention_mask = attention_mask.to(device, non_blocking=True)
                # Move labels para o device
                labels = labels.to(device, non_blocking=True)

                # Executa forward e obtém a perda de linguagem
                loss, _ = model(input_ids, attention_mask, labels=labels)

                # Zera gradientes de forma eficiente (set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                # Backpropaga a perda para acumular gradientes
                loss.backward()
                # Aplica clipping de gradiente para estabilidade (explode grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Atualiza os pesos com o otimizador
                optimizer.step()

                # Acumula a perda da iteração para média por época
                running_loss += loss.item()

                # Logging ocasional por batch (a cada 100 passos)
                if i % 100 == 0:
                    # Exibe informações de progresso no console
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}] "
                        f"Step [{i+1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )
                    # Registra a perda por batch no MLflow (step = global_step)
                    mlflow.log_metric("train_batch_loss", loss.item(), step=global_step)

                # Incrementa o step global após cada batch
                global_step += 1

            # Calcula a perda média da época
            epoch_avg_loss = running_loss / len(train_loader)
            # Exibe a perda média da época
            print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {epoch_avg_loss:.4f}")
            # Registra a perda média da época no MLflow (step = epoch)
            mlflow.log_metric("train_epoch_loss", epoch_avg_loss, step=epoch)

        # Ao final do treino, salva apenas o GPT-2 ajustado (fine-tuned) como artefato
        mlflow.pytorch.log_model(model.gpt, "fine_tuned_gpt_model")

        # Executa avaliação final no conjunto de teste
        evaluate_model(model)

# Ponto de entrada: executa o treinamento quando o arquivo é chamado como script
if __name__ == "__main__":
    train_model()
