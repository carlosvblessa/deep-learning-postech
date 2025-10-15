# POSTECH FCD - Modulo 4: Deep Learning

Reuniao de roteiros, notebooks, scripts e materiais de apoio utilizados nas aulas do modulo de Deep Learning do curso de Formacao em Ciencia de Dados (FCD) da POSTECH. O objetivo e oferecer exemplos completos que conectam fundamentos teoricos, implementacoes com PyTorch e boas praticas de monitoramento e productizacao.

## Estrutura
- `LICENSE`: licenca do conteudo.
- `src/`: codigo-fonte organizado por topico. Principais pastas:
  - `activation_functions/`: demonstracoes de sigmoid, tanh, relu e backpropagation com rastreio via MLflow.
  - `architectures/`: redes profundas agrupadas por categoria (`cnns`, `rnns`, `gans`, `transformers`) utilizando dados sinteticos para treino e metricas registradas com MLflow.
  - `optimization/`: tecnicas como transfer learning, regularizacao, pruning, quantization, monitoreo e scripts de operacao para levada a producao.
  - `productization/`: API Flask para inferencia com modelo ResNet, monitoramento via Prometheus e artefatos de deploy (`dockerfile`, `prometheus.yaml`, modelo salvo).
  - `topics/` e `use_cases/`: exemplos adicionais (VAE, pipelines de GenAI) prontos para demonstracoes em aula.
  - `data/` e `generate_ft_data_azure.py`: conjunto de dados de fine-tuning em formato JSONL e script para gerar novas amostras para Azure AI Content Safety.

## Conteudo das Aulas
- **Aula 01 – Algoritmos classicos e Perceptron:** revisita perceptron de camada unica, compara com regressao linear e k-NN, discute ajuste de pesos, espaco de hipoteses, vies/variancia e limita o que redes rasas conseguem aprender.
- **Aula 02 – Teoria das Redes Neurais Profundas:** aprofunda em redes multicamadas, fundamentos probabilisticos, funcoes de ativacao, espacos latentes, backpropagation e regularizacao (L1/L2, equivariancia), com exemplos de baixo nivel em C++.
- **Aula 03 – Arquiteturas de Redes Neurais Profundas:** apresenta arquiteturas modernas (CNNs, RNNs/LSTM/GRU, GANs, transformers, difusores), sua evolucao historica e cuidados de otimizacao e produtizacao.
- **Aula 04 – Tecnicas de Aplicacao:** conecta redes profundas a cenarios supervisionados, nao supervisionados, semi-supervisionados e de reforco, cobrindo monitoramento de drift, escalabilidade e práticas de integracao continua.
- **Aula 05 – Casos de Uso:** explora aplicacoes em PLN, visao computacional, IA generativa e integracoes transacionais (detecao de fraude, recomendacao, automacao), destacando requisitos operacionais para rodar modelos em producao.

## Requisitos e Preparacao
- Python 3.11 com `pip` atualizado
- GPU opcional para acelerar redes profundas e transformers

### Setup (venv + pip)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> Se preferir outro gerenciador (como Poetry ou UV), utilize `src/pyproject.toml` como referencia para as mesmas dependencias.

## Executando os Exemplos
- Funcoes de ativacao: `python src/activation_functions/relu.py`
- Arquiteturas de redes: `python src/architectures/cnns/alexnet.py`, `python src/architectures/transformers/bert.py`, etc. Os scripts geram dados artificiais, treinam o modelo e registram metricas no MLflow.
- Tecnicas de otimizacao: `python src/optimization/transfer_learning.py` (usa imagens de `src/optimization/imagens`), `python src/optimization/quantization.py`, entre outros.
- Productizacao: `python src/productization/app.py` sobe uma API Flask em `http://localhost:5000/predict` e expõe metricas em `/metrics`. Ajuste o caminho do modelo salvo se mover arquivos.
- Geracao de dataset para Azure: `python src/generate_ft_data_azure.py` cria `src/data/fine_tuning_data.jsonl` com pares pergunta-resposta para fine-tuning de content safety.

### Visualizacao de Experimentos
Scripts registram metricas e parametros no MLflow padrao (`./mlruns`). Para acompanhar a evolucao:
```bash
mlflow ui --port 5001
```
Em seguida acesse `http://localhost:5001`.

## Boas Praticas
- Reduza `num_epochs` ou `num_samples` nos scripts antes de demonstracoes ao vivo se precisar de execucoes mais rapidas.
- Certifique-se de que os downloads do Hugging Face (BERT, T5) e do MNIST estejam disponiveis no ambiente ou habilite acesso a rede.
- Para reproducao em GPU, verifique a instalacao correta de PyTorch com CUDA.
- Utilize os dashboards do Prometheus (com `src/productization/prometheus.yaml`) para acompanhar latencia, contagem de inferencias e acuracia simulada.

## Licenca
Consulte `LICENSE` para termos de uso do material.
