# Sistemas Conexionistas — Previsão de Energia Solar e Preço de Voos

Trabalho prático da unidade curricular **Sistemas Conexionistas para Aprendizagem Automática**, Licenciatura em Ciência de Dados — Universidade do Minho, 3º ano, 1º semestre (2025/2026).

Duas tarefas de Machine Learning com redes neuronais:

## Parte A — Previsão de Energia Solar (competição Kaggle)

Classificação horária da produção de energia solar (painéis em Braga, set/2021–abr/2023) em classes ordinais (`None → Very_High`), no âmbito da competição [Kaggle SCAA-2526](https://www.kaggle.com/c/scaa-2526).

- Dados reais de energia + meteorologia, treino 2021–2022, teste 2023 (sem labels)
- Comparação entre arquiteturas: **Feedforward**, **RNN/LSTM/GRU** (modelos sequenciais) e **CNN 1D** com janela temporal de 24 horas
- Modelo final: CNN 1D — melhor validation accuracy (~0.90)
- Desafios tratados: mudança de distribuição temporal entre treino/teste, desbalanceamento de classes

## Parte B — Previsão do Preço de Voos

Regressão do preço de bilhetes de avião (dataset `Clean_Dataset.csv`), usando uma rede neuronal com **embeddings** para variáveis categóricas (companhia aérea, origem, destino, classe, horário) concatenados com variáveis numéricas (duração, escalas, dias de antecedência).

- Pré-processamento: codificação por índices inteiros + categoria `UNK` para valores não vistos (evita one-hot / alta dimensionalidade)
- Arquitetura: Embeddings + numéricos → Concatenate → Dense (ReLU) → Dropout → saída linear
- Comparação de regularização: Dropout 0.3 (MAE ≈ 3094) teve melhor desempenho que Dropout 0.5 (MAE ≈ 3193)

## Autoria

Projeto de grupo (Grupo 11):

- João Pedro / João Duarte
- **Eduarda Ferreira**
- Rafael Ferreira

## Estrutura do repositório

```
├── notebooks/
│   ├── parteA.ipynb          # Previsão de energia solar (classificação)
│   └── parteB.ipynb          # Previsão de preço de voos (regressão)
├── data/
│   └── flights/              # Dados da Parte B (voos): dataset limpo, splits gold, mappings
├── models/
│   ├── best_model_flights.keras
│   └── best_model_flights_v2.keras
├── docs/
│   ├── relatorio.pdf 
│   ├── enunciado.pdf         # Enunciado do trabalho
└── README.md
```

## Dados e resultados (Parte A)

- `data/solar/` — dados de energia e meteorologia (set/2021 – abr/2023)
- `notebooks/` — notebooks de experimentação (early stopping, class weights, SMOTE, grid search)
- `results/` — métricas por arquitetura (CNN, RNN, RL, API) e histórico de submissões ao Kaggle, incluindo experiências com class weights, SMOTE, focal loss, label smoothing, abordagem em duas etapas e Reinforcement Learning (DQN)

## Stack

Python, TensorFlow/Keras, pandas, scikit-learn
