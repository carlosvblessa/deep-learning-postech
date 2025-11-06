# Caminho do arquivo (informativo)
# src/optimization/monitoracao.py

# Importa utilitários de SO
import os
# Acesso a argumentos/saída padrão
import sys
# Utilitário de tempo (duração do job)
import time
# Logging estruturado
import logging
# Manipulação de caminhos de forma portátil
from pathlib import Path
# Tipagem para classes de Loss/Optimizer passadas como parâmetro
from typing import Type

# NumPy para utilidades numéricas
import numpy as np
# Núcleo do PyTorch
import torch
# Camadas e módulos de rede neural
import torch.nn as nn
# Otimizadores do PyTorch
import torch.optim as optim
# Tipo base de perdas ponderadas (para receber a classe de loss)
from torch.nn.modules.loss import _WeightedLoss
# Tipo base de otimizadores (para receber a classe de optimizer)
from torch.optim.optimizer import Optimizer
# Datasets, modelos e transforms do torchvision
from torchvision import datasets, models, transforms
# DataLoader para batching
from torch.utils.data import DataLoader

# MLflow para experiment tracking
import mlflow
# Submódulo para salvar/logar modelos PyTorch
import mlflow.pytorch
# Utilitário para inferir assinatura (schema) de entrada/saída do modelo
from mlflow.models.signature import infer_signature


# Configura logger básico (nome 'train' + saída no stdout)
logger = logging.getLogger("train")
# Evita handlers duplicados se este módulo for importado mais de uma vez
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))
# Nível de log: INFO (muda para DEBUG se quiser mais verbosidade)
logger.setLevel(logging.INFO)


# Cria uma ResNet18 pré-treinada compatível com versões novas/antigas do torchvision
def _get_resnet18_pretrained() -> nn.Module:
    """
    Cria uma ResNet18 pré-treinada de forma compatível com versões novas/antigas da torchvision.
    """
    # Tenta API nova (torchvision >= 0.13) com enum de pesos
    try:
        from torchvision.models import ResNet18_Weights  # type: ignore
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Se falhar, usa a API antiga (pretrained=True)
    except Exception:
        model = models.resnet18(pretrained=True)
    # Retorna o backbone carregado
    return model


# Função principal: orquestra dados, modelo, treino, MLflow e salvamento
def main(
    version: int,
    criterion_class: Type[_WeightedLoss],
    optimizer_class: Type[Optimizer],
) -> nn.Module:
    # Marca o início do job (timestamp)
    start_time = time.time()
    # Loga início do job
    logger.info("Starting job at %.0f", start_time)

    # Define raiz do projeto (diretório atual)
    proj_root = Path.cwd()
    # Caminho do dataset (espera subpastas train/ e val/)
    data_dir = proj_root / "src" / "optimization" / "imagens"   # dataset: train/, val/
    # Verifica se as pastas existem e falha cedo se não existirem
    assert (data_dir / "train").exists() and (data_dir / "val").exists(), (
        f"Pastas de dados não encontradas em {data_dir}. "
        f"Esperado: {data_dir/'train'} e {data_dir/'val'}"
    )

    # Hiperparâmetros principais
    num_classes = 2
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3

    # Diretório local para pesos/artefatos auxiliares (versionado por parâmetros)
    artifact_dir = proj_root / "src" / "optimization" / "artifacts" / f"v{version}_LR_{learning_rate}_bs_{batch_size}"
    # Garante que a pasta exista
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Caminho do arquivo de pesos a salvar no fim
    weights_path = artifact_dir / f"model_monitoring_v{version}.pth"

    # Define transforms de treino/val (normalização ImageNet)
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Cria datasets a partir das pastas (classe = nome da subpasta)
    image_datasets = {
        split: datasets.ImageFolder(root=(data_dir / split), transform=data_transforms[split])
        for split in ["train", "val"]
    }

    # Configura workers e pin_memory conforme o ambiente
    num_workers = min(4, (os.cpu_count() or 1))
    pin_memory = torch.cuda.is_available()
    # Constrói DataLoaders para treino/val
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        for split in ["train", "val"]
    }
    # Guarda tamanhos dos datasets (para métricas normalizadas)
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    # Mapeamento classe->índice (útil para interpretação)
    class_to_idx = image_datasets["train"].class_to_idx

    # Constrói modelo pré-treinado (ResNet18)
    model = _get_resnet18_pretrained()
    # Lê dimensão da FC e troca por uma nova de 2 classes
    num_ftrs = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = nn.Linear(num_ftrs, num_classes)  # type: ignore[attr-defined]

    # Congela backbone e deixa apenas a FC treinável (transfer learning rápido)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():  # type: ignore[attr-defined]
        p.requires_grad = True

    # Seleciona device CPU (pode ser alterado para GPU se disponível)
    device = torch.device("cpu")
    # Move o modelo para o device selecionado
    model = model.to(device)

    # Instancia a loss a partir da classe recebida (ex.: CrossEntropyLoss)
    criterion = criterion_class()
    # Instancia o otimizador a partir da classe recebida (ex.: Adam) apenas na FC
    optimizer = optimizer_class(model.fc.parameters(), lr=learning_rate)  # type: ignore[arg-type]

    # Define função interna de treino/val com logging e checkpoint simples
    def train_model(m: nn.Module, loss_fn: nn.Module, opt: Optimizer, epochs: int = 10) -> nn.Module:
        # Melhor acurácia de validação observada
        best_acc = 0.0
        # Snapshot de melhores pesos
        best_state = None

        # Loop de épocas
        for epoch in range(epochs):
            # Cabeçalho da época
            logger.info("Epoch %d/%d", epoch + 1, epochs)
            logger.info("-" * 20)

            # Fases de treino e validação
            for phase in ["train", "val"]:
                # Alterna modo do modelo
                if phase == "train":
                    m.train()
                else:
                    m.eval()

                # Acumuladores
                running_loss = 0.0
                running_corrects = 0

                # Loop de batches
                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    # Move batch para o device
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Zera gradientes
                    opt.zero_grad()

                    # Liga grad apenas no treino
                    with torch.set_grad_enabled(phase == "train"):
                        # Forward → logits
                        outputs = m(inputs)
                        # Predição por argmax
                        _, preds = torch.max(outputs, 1)
                        # Calcula perda
                        loss = loss_fn(outputs, labels)

                        # Backprop + update só no treino
                        if phase == "train":
                            loss.backward()
                            opt.step()

                    # Acumula perda ponderada e acertos
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)

                    # Log de depuração no primeiro batch (evita spam)
                    if batch_idx == 0:
                        logger.debug(
                            "[%s] batch0 shapes: inputs=%s labels=%s outputs=%s",
                            phase,
                            tuple(inputs.shape),
                            tuple(labels.shape),
                            tuple(outputs.shape),
                        )

                # Normaliza métricas por amostra
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

                # Loga métricas no console
                logger.info("%s Loss: %.4f Acc: %.4f", phase, epoch_loss, epoch_acc)
                # Loga métricas no MLflow (step = epoch)
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_acc", epoch_acc, step=epoch)

                # Checkpoint: guarda melhor estado pelo val_acc
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}

        # Restaura melhor estado (se houver)
        if best_state is not None:
            m.load_state_dict(best_state)

        # Loga resumo final
        logger.info("Treinamento completo. Melhor val_acc: %.4f", best_acc)
        # Retorna modelo (com melhores pesos)
        return m

    # Configura o tracking do MLflow para usar a pasta local ./mlruns (via file://)
    mlflow.set_tracking_uri(f"file://{(proj_root / 'mlruns').as_posix()}")
    # Define/seleciona experimento
    exp_name = "Monitoração em Tempo de Treinamento com ResNet"
    mlflow.set_experiment(exp_name)

    # Abre um run do MLflow (agrupa métricas/artefatos)
    with mlflow.start_run(run_name=f"Experimento_Monitorado_v{version}"):
        # Define tags informativas sobre a execução
        mlflow.set_tag("version", version)
        mlflow.set_tag("optimizer", optimizer_class.__name__)
        mlflow.set_tag("criterion", criterion_class.__name__)
        # Loga hiperparâmetros principais do job
        mlflow.log_params(
            {
                "num_classes": num_classes,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
            }
        )
        # Salva mapeamento de classes localmente e loga como artefato
        (artifact_dir / "class_to_idx.txt").write_text(str(class_to_idx), encoding="utf-8")
        mlflow.log_artifact(str(artifact_dir / "class_to_idx.txt"), artifact_path="local_artifacts")

        # Executa treinamento (com checkpoints e logs por época)
        model = train_model(model, criterion, optimizer, epochs=num_epochs)

        # Prepara exemplo e assinatura para logar o modelo no MLflow
        # (usa entrada sintética caso não capture um batch real aqui)
        model.eval()
        example_np = np.random.rand(1, 3, 224, 224).astype("float32")
        with torch.no_grad():
            _out = model(torch.from_numpy(example_np).to(device))
            out_np = _out.detach().cpu().numpy()

        # Infere assinatura entrada→saída
        signature = infer_signature(example_np, out_np)
        # Loga o modelo PyTorch no MLflow com exemplo e assinatura
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="pytorch_model",
            input_example=example_np,
            signature=signature,
        )

        # Salva pesos locais em .pth
        torch.save(model.state_dict(), weights_path)
        # Loga o .pth como artefato
        mlflow.log_artifact(str(weights_path), artifact_path="local_artifacts")

        # (Opcional) Loga toda a pasta de artefatos locais para inspeção
        mlflow.log_artifacts(str(artifact_dir), artifact_path="local_artifacts_all")

        # Marca fim do job e loga duração
        end_time = time.time()
        logger.info("Job finished at %.0f | Elapsed: %.1fs", end_time, end_time - start_time)
        # Mostra run_id e URI de artefatos úteis para localizar o run
        logger.info("Run ID: %s | Artifacts: %s", mlflow.active_run().info.run_id, mlflow.get_artifact_uri())

    # Retorna o modelo treinado (melhores pesos)
    return model


# Execução direta do módulo (exemplo simples: versão 1, CrossEntropy + Adam)
if __name__ == "__main__":
    # Chama a main com classes de loss/optimizer passadas por parâmetro
    main(1, nn.CrossEntropyLoss, optim.Adam)
