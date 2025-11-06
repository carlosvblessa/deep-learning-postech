# Caminho do arquivo (informativo)
# src/optimization/transfer_learning.py

# Importa utilidades do sistema operacional (não obrigatório aqui, mas comum em scripts)
import os
# Importa Path para caminhos portáveis (Windows/Linux/Mac)
from pathlib import Path
# Tipos para anotações (dicionários e tuplas)
from typing import Dict, Tuple

# Importa PyTorch principal (tensores e execução)
import torch
# Importa módulos de camadas e redes neurais
import torch.nn as nn
# Importa otimizadores do PyTorch
import torch.optim as optim
# Importa DataLoader para batching
from torch.utils.data import DataLoader
# Importa datasets utilitários, modelos pré-treinados e transforms de imagem
from torchvision import datasets, models, transforms

# MLflow para experiment tracking (parâmetros, métricas e artefatos)
import mlflow
# Submódulo para logar/salvar modelos PyTorch no MLflow
import mlflow.pytorch
# Utilitário do MLflow para inferir a assinatura (schema) de entrada/saída do modelo
from mlflow.models.signature import infer_signature


# Define função para construir DataLoaders e metadados de datasets/classes
def build_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[str, int]]:
    """
    Cria dataloaders de train/val e retorna também tamanhos e número de classes.
    """
    # Define pipelines de pré-processamento/augmentação para treino e validação
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),            # crop aleatório para 224x224
            transforms.RandomHorizontalFlip(),           # espelhamento horizontal
            transforms.ToTensor(),                       # para tensor [0,1]
            transforms.Normalize([0.485, 0.456, 0.406],  # normalização padrão ImageNet
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),                      # resize curto para 256
            transforms.CenterCrop(224),                  # crop central para 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    # Cria datasets a partir da estrutura de pastas: data_dir/train e data_dir/val
    image_datasets = {
        split: datasets.ImageFolder(root=(data_dir / split).resolve(),
                                    transform=data_transforms[split])
        for split in ["train", "val"]
    }

    # Cria DataLoaders com pin_memory automático se houver CUDA disponível
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=True,                        # mistura amostras por época
            num_workers=num_workers,             # workers para I/O paralelo
            pin_memory=torch.cuda.is_available() # otimiza transferências H->D em CUDA
        )
        for split in ["train", "val"]
    }

    # Calcula o número de amostras em cada split (para métricas médias)
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    # Obtém a quantidade de classes a partir do dataset de treino
    num_classes = len(image_datasets["train"].classes)

    # Retorna os dataloaders, os tamanhos e um dict com num_classes
    return dataloaders, dataset_sizes, {"num_classes": num_classes}


# Constrói a ResNet-18 pré-treinada e ajusta a camada final para o número de classes
def build_model(num_classes: int) -> nn.Module:
    """
    Carrega ResNet-18 com pesos pré-treinados e ajusta a FC final.
    Suporta tanto API nova (weights=...) quanto antiga (pretrained=True).
    """
    # Tenta usar a API nova (torchvision >= 0.13) com enum de pesos
    try:
        weights = models.ResNet18_Weights.DEFAULT  # torchvision >= 0.13
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback para versões antigas que usam 'pretrained=True'
        model = models.resnet18(pretrained=True)

    # Obtém a dimensionalidade de entrada da FC e substitui pela cabeça nova
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    # Retorna a rede pronta para treino
    return model


# Executa uma época de treinamento e acumula soma de perdas e acertos
def train_one_epoch(model, dataloader, device, criterion, optimizer):
    # Coloca o modelo em modo treino (ativa dropout/batchnorm)
    model.train()
    # Acumuladores de perda e acertos ponderados pelo tamanho do batch
    running_loss = 0.0
    running_corrects = 0

    # Itera sobre os batches do DataLoader de treino
    for inputs, labels in dataloader:
        # Tamanho do batch atual (para ponderar perda)
        bsz = inputs.size(0)
        # Move dados para o device (CPU ou CUDA)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zera gradientes a cada iteração
        optimizer.zero_grad()
        # Forward: logits [B, num_classes]
        outputs = model(inputs)
        # Predições como argmax por classe
        _, preds = torch.max(outputs, 1)
        # Perda de classificação
        loss = criterion(outputs, labels)
        # Backprop da perda
        loss.backward()
        # Atualiza pesos
        optimizer.step()

        # Acumula perda ponderada e acertos do batch
        running_loss += loss.item() * bsz
        running_corrects += torch.sum(preds == labels)

    # Retorna somas (serão normalizadas pelo chamador)
    return running_loss, running_corrects


# Desativa gradientes e avalia no split de validação, acumulando métricas
@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    # Modo avaliação (desativa dropout, usa running stats da batchnorm)
    model.eval()
    # Acumuladores de perda e acertos
    running_loss = 0.0
    running_corrects = 0

    # Itera sobre os batches do DataLoader de validação
    for inputs, labels in dataloader:
        # Tamanho do batch
        bsz = inputs.size(0)
        # Move tensores para o device
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward e cálculo de perda
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Acumula perda ponderada e acertos
        running_loss += loss.item() * bsz
        running_corrects += torch.sum(preds == labels)

    # Retorna somas acumuladas (normalização fica fora)
    return running_loss, running_corrects


# Função principal que orquestra dados, modelo, treino, logging e salvamento
def main():
    # Define diretório base contendo 'train/' e 'val/'
    data_dir = Path.cwd() / "src" / "optimization" / "imagens"
    # Tamanho do batch
    batch_size = 32
    # Número de épocas de treinamento
    num_epochs = 10
    # Taxa de aprendizado do otimizador
    learning_rate = 1e-3
    # Número de workers para DataLoader (I/O)
    num_workers = 4

    # Observação: sem set_tracking_uri/ set_experiment → usa './mlruns' (Default) por padrão
    # Para UI: execute 'mlflow ui --backend-store-uri ./mlruns' no diretório do projeto

    # Seleciona o device (aqui fixado em CPU)
    device = torch.device("cpu")

    # Constrói DataLoaders, tamanhos dos datasets e metadados (num_classes)
    dataloaders, dataset_sizes, meta = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Cria a ResNet-18 ajustada para o número de classes e move para o device
    model = build_model(num_classes=meta["num_classes"]).to(device)
    # Define a loss para classificação multiclasse
    criterion = nn.CrossEntropyLoss()
    # Otimizador (apenas a cabeça 'fc' neste exemplo)
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Abre um run do MLflow; aparecerá no experimento "Default"
    with mlflow.start_run(run_name="resnet18_transfer_learning"):
        # Loga hiperparâmetros e configurações principais do treino
        mlflow.log_params({
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
            "device": str(device),
            "data_dir": str(data_dir.resolve()),
            "num_classes": meta["num_classes"],
        })

        # Rastreia melhor acurácia de validação e snapshot dos melhores pesos
        best_acc = 0.0
        best_state = None

        # Loop de épocas
        for epoch in range(num_epochs):
            # Executa uma época de treino e obtém somas acumuladas
            tr_loss_sum, tr_corr = train_one_epoch(
                model, dataloaders["train"], device, criterion, optimizer
            )
            # Normaliza perda e acurácia pelo tamanho do dataset de treino
            tr_loss = tr_loss_sum / dataset_sizes["train"]
            tr_acc = tr_corr.double().item() / dataset_sizes["train"]

            # Avalia no split de validação e normaliza métricas
            va_loss_sum, va_corr = evaluate(
                model, dataloaders["val"], device, criterion
            )
            va_loss = va_loss_sum / dataset_sizes["val"]
            va_acc = va_corr.double().item() / dataset_sizes["val"]

            # Prints de acompanhamento no console (saída esperada na interface)
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f}")
            print(f"val   Loss: {va_loss:.4f} Acc: {va_acc:.4f}")
            # Registro de métricas no MLflow (aparecem no run atual)
            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            mlflow.log_metric("train_acc", tr_acc, step=epoch)
            mlflow.log_metric("val_loss", va_loss, step=epoch)
            mlflow.log_metric("val_acc", va_acc, step=epoch)

            # Atualiza melhor modelo se a acurácia de validação melhorou
            if va_acc > best_acc:
                best_acc = va_acc
                # Clona pesos para CPU (garante portabilidade ao salvar)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Mensagem final com a melhor acurácia de validação da execução
        print(f"Treinamento completo. Melhor val_acc: {best_acc:.4f}")

        # Se houve um melhor snapshot, restaura os pesos no modelo
        if best_state is not None:
            model.load_state_dict(best_state)

        # Prepara um exemplo real do loader de validação para logar no MLflow
        try:
            # Toma 1 amostra do DataLoader de validação
            sample_inputs, _ = next(iter(dataloaders["val"]))
            example_t = sample_inputs[:1].to(device)  # forma [1, 3, 224, 224]
        except StopIteration:
            # Se por algum motivo o val estiver vazio, usa ruído com a mesma forma
            example_t = torch.randn(1, 3, 224, 224, device=device)

        # Coloca o modelo em avaliação para gerar saída do exemplo
        model.eval()
        with torch.no_grad():
            out_t = model(example_t)

        # Converte tensores para NumPy: o MLflow não aceita Tensor em input_example
        example_np = example_t.detach().cpu().numpy()
        out_np = out_t.detach().cpu().numpy()
        # Infere a assinatura (schemas de entrada e saída) para o modelo
        signature = infer_signature(example_np, out_np)

        # Loga o modelo no MLflow com exemplo e assinatura (artefato 'pytorch_model')
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_model",
            input_example=example_np,
            signature=signature,
        )

        # Também salva localmente os pesos em um arquivo .pth (opcional)
        torch.save(model.state_dict(), "model_transfer_learning.pth")
        # E registra esse arquivo como artefato do run
        mlflow.log_artifact("model_transfer_learning.pth")


# Bloco padrão de execução direta do script
if __name__ == "__main__":
    mlflow.set_experiment("Transferencia de Aprendizado com ResNet pretreinada")    
    # Chama a função principal; imprime no console e loga no MLflow (./mlruns/Default)
    main()
