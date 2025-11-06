# Caminho do arquivo (informativo)
# src/architectures/transformers/t5.py

# Importa funções matemáticas utilitárias (p.ex., exp/perplexidade se necessário)
import math
# Importa o gerador pseudoaleatório padrão do Python
import random
# Importa NumPy para operações auxiliares com arrays e seeds
import numpy as np
# Importa o núcleo do PyTorch (tensores e kernel numérico)
import torch
# Importa módulos de camadas/containers de rede; aqui só para o wrapper
import torch.nn as nn  # wrapper T5Model
# Importa DataLoader/TensorDataset para pipeline de dados
from torch.utils.data import DataLoader, TensorDataset
# Importa classes do ecossistema T5 e o scheduler linear com warmup
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    get_linear_schedule_with_warmup,
)
# Importa AdamW (otimizador padrão para Transformers)
from torch.optim import AdamW
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
# Fixa a semente do PyTorch
torch.manual_seed(SEED)
# Se estiver em CUDA, fixa as sementes das GPUs também
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Nome do checkpoint base do T5
model_name = "t5-small"
# Número de épocas de treinamento
num_epochs = 5
# Tamanho do minibatch
batch_size = 16
# Taxa de aprendizado
learning_rate = 1e-4
# Comprimento (máximo) das sequências de entrada/saída
sequence_length = 20
# Quantidade de amostras sintéticas
num_samples = 500
# Proporção de passos para warmup do scheduler
warmup_ratio = 0.06   # ~6% dos passos para aquecimento
# Valor máximo da norma do gradiente (clipping)
max_grad_norm = 1.0

# Cria o tokenizer do T5
# Define explicitamente o modo do tokenizer para evitar o aviso de "legacy behaviour".
# Se quiser o novo comportamento, mude legacy=True -> legacy=False.
try:
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
except TypeError:
    # Para versões mais antigas do transformers que não aceitam 'legacy'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

# Carrega a configuração do T5 e desativa o cache de chaves/valores (past_key_values)
config = T5Config.from_pretrained(model_name)
config.use_cache = False  # evita aviso de past_key_values legacy

# Carrega o modelo T5 para geração condicional e move para o device
t5 = T5ForConditionalGeneration.from_pretrained(model_name, config=config).to(device)
# Garante em runtime que o cache não será usado (redundante, mas explícito)
t5.config.use_cache = False  # redundante, mas garante em runtime

# Define um gerador de dados artificiais para uma tarefa sintética de tradução
def generate_artificial_data(num_samples: int, sequence_length: int):
    # Tarefa sintética de tradução (T5 usa prefixos de tarefa)
    input_sentences = [
        "translate English to French: " + " ".join(["word"] * sequence_length)
        for _ in range(num_samples)
    ]
    # Cria sentenças alvo simples (palavra francesa repetida)
    target_sentences = [" ".join(["mot"] * sequence_length) for _ in range(num_samples)]

    # Tokeniza entradas com padding/truncation até sequence_length
    input_enc = tokenizer(
        input_sentences,
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    # Tokeniza alvos com padding/truncation até sequence_length
    target_enc = tokenizer(
        target_sentences,
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )

    # Extrai ids de entrada e máscara de atenção
    input_ids = input_enc["input_ids"]
    attention_mask = input_enc["attention_mask"]
    # Extrai labels (ids do alvo)
    labels = target_enc["input_ids"]

    # Mascara o padding nas labels com -100 (ignore_index da CrossEntropy)
    labels[target_enc["attention_mask"] == 0] = -100

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

# Cria o TensorDataset de treino (inputs, máscara, labels)
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
# Cria o TensorDataset de teste (inputs, máscara, labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

# Habilita pin_memory quando o device for CUDA (acelera cópias host->GPU)
pin_mem = device == "cuda"
# DataLoader de treino com embaralhamento
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem, num_workers=2
)
# DataLoader de teste sem embaralhamento
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem, num_workers=2
)

# Define um wrapper simples para padronizar a interface (loss, logits)
class T5Model(nn.Module):
    # Construtor recebe uma instância de T5 já carregada
    def __init__(self, t5: T5ForConditionalGeneration):
        super().__init__()
        self.t5 = t5

    # Forward compatível com treino/val: retorna (loss, logits)
    def forward(self, input_ids, attention_mask, labels):
        # use_cache=False para evitar aviso de past_key_values
        out = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        # Retorna a perda e os logits token-a-token
        return out.loss, out.logits

# Define a rotina de avaliação no conjunto de teste
def evaluate_model(model: T5Model):
    # Coloca o modelo em modo de avaliação (desativa dropout, etc.)
    model.eval()
    # Acumulador de perda total
    total_loss = 0.0
    # Contador de minibatches
    n_batches = 0

    # Desativa gradientes para acelerar a inferência
    with torch.inference_mode():
        # Itera sobre os batches do DataLoader de teste
        for input_ids, attention_mask, labels in test_loader:
            # Move tensores para o device
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Calcula a perda do batch; logits não são usados aqui
            loss, _ = model(input_ids, attention_mask, labels)
            # Acumula a perda
            total_loss += loss.item()
            # Incrementa número de batches
            n_batches += 1

    # Calcula a perda média por minibatch
    avg_loss = total_loss / max(1, n_batches)
    # Exibe métrica de avaliação no console
    print(f"Test Loss: {avg_loss:.4f}")
    # Registra a métrica no MLflow
    mlflow.log_metric("test_loss", avg_loss)

# Define a rotina principal de treinamento
def train_model():
    # Envolve o T5 no wrapper para interface (loss, logits)
    model = T5Model(t5).to(device)

    # Configura AdamW e o scheduler linear com warmup
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # Total de passos de treinamento (épocas × batches)
    num_training_steps = num_epochs * len(train_loader)
    # Passos de warmup conforme a proporção definida
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    # Scheduler linear com warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # Define/seleciona o experimento no MLflow
    mlflow.set_experiment("T5 Artificial Data Generation")
    # Abre um run para registrar parâmetros, métricas e artefatos
    with mlflow.start_run():
        # Log de hiperparâmetros e configs relevantes
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("warmup_ratio", warmup_ratio)
        mlflow.log_param("max_grad_norm", max_grad_norm)
        mlflow.log_param("tokenizer_legacy_flag", "True")  # estamos usando legacy=True explicitamente
        mlflow.log_param("use_cache", False)

        # Contador global de passos (para métricas por batch)
        global_step = 0
        # Loop principal de épocas
        for epoch in range(num_epochs):
            # Modo treino
            model.train()
            # Acumulador de perda por época
            running_loss = 0.0

            # Itera sobre os batches do DataLoader de treino
            for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
                # Move tensores para o device
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Forward com labels (o modelo calcula a perda internamente)
                loss, _ = model(input_ids, attention_mask, labels)

                # Zera gradientes de forma eficiente
                optimizer.zero_grad(set_to_none=True)
                # Backpropaga a perda
                loss.backward()
                # Clipping de gradiente para estabilidade
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # Atualiza os pesos
                optimizer.step()
                # Avança o scheduler um passo
                scheduler.step()

                # Acumula a perda do batch
                running_loss += loss.item()

                # Logging ocasional por batch
                if i % 100 == 0:
                    # Exibe progresso no console
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}] "
                        f"Step [{i+1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )
                    # Registra a perda por batch no MLflow
                    mlflow.log_metric("train_batch_loss", loss.item(), step=global_step)

                # Incrementa o step global
                global_step += 1

            # Calcula a perda média da época
            epoch_avg_loss = running_loss / len(train_loader)
            # Exibe a perda média no console
            print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {epoch_avg_loss:.4f}")
            # Loga a perda média da época no MLflow (step = epoch)
            mlflow.log_metric("train_epoch_loss", epoch_avg_loss, step=epoch)

        # Ao final do treino, salva o T5 ajustado (fine-tuned) como artefato
        mlflow.pytorch.log_model(model.t5, "fine_tuned_t5_model")

        # Executa avaliação final no conjunto de teste
        evaluate_model(model)

# Ponto de entrada: executa o treinamento quando chamado como script
if __name__ == "__main__":
    train_model()
