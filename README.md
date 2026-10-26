# Industrial Predictive Maintenance

Protótipo de Machine Learning para análise de dados de máquinas e classificação de registros associados a falhas, utilizando o dataset público **AI4I 2020 Predictive Maintenance Dataset**.

O projeto explora um fluxo básico de Machine Learning com Python: análise exploratória, preparação dos dados, treinamento de um modelo, avaliação de métricas e classificação de novas entradas pelo terminal.

> **Aviso:** este projeto é experimental e educacional. O dataset é sintético, e os resultados não comprovam a capacidade de prever falhas futuras em máquinas industriais reais.

## Objetivos

- Explorar e compreender os dados de manutenção preditiva.
- Investigar a relação entre características operacionais e registros de falha.
- Treinar um classificador utilizando Random Forest.
- Avaliar o desempenho com métricas adequadas a um conjunto de dados desbalanceado.
- Investigar falsos positivos e falsos negativos.
- Testar classificações a partir de dados informados pelo usuário.

## Tecnologias utilizadas

- **Python** — linguagem principal.
- **Pandas** — manipulação e análise de dados.
- **Scikit-learn** — pré-processamento, treinamento e avaliação do modelo.
- **Joblib** — persistência e carregamento do modelo treinado.

## Dataset

O projeto utiliza o [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), disponibilizado pelo UCI Machine Learning Repository.

O conjunto contém **10.000 registros e 14 colunas**, incluindo:

- Tipo da máquina.
- Temperatura do ar e do processo.
- Velocidade de rotação.
- Torque.
- Desgaste da ferramenta.
- Indicador de falha da máquina.

A variável-alvo é `Machine failure`, que indica se o registro está associado a uma falha.

A distribuição original apresenta forte desbalanceamento: aproximadamente **96,61% dos registros não apresentam falha e 3,39% apresentam falha**.

### Preparação dos dados

O pipeline utiliza `OneHotEncoder` para transformar a variável categórica em valores numéricos e mantém as variáveis numéricas no fluxo de processamento.

As colunas identificadoras, a variável-alvo e os indicadores específicos dos tipos de falha são excluídos das entradas do modelo.

Os dados são separados em 80% para treinamento e 20% para teste, com estratificação das classes. O treinamento utiliza `RandomForestClassifier`, com 200 árvores e `class_weight="balanced"` para lidar com o desbalanceamento.

## Estrutura do projeto

```text
industrial-predictive-maintenance/
├── data/
│   └── ai4i2020.csv
├── models/
│   └── failure_model.joblib
├── src/
│   ├── explore.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── .gitignore
└── requirements.txt
```

- `explore.py` — apresenta dimensões, colunas, tipos de dados, valores ausentes e distribuição das classes.
- `train.py` — prepara os dados, treina o modelo, apresenta métricas e salva o modelo treinado.
- `predict.py` — recebe características de uma máquina pelo terminal e apresenta a classificação e a pontuação estimada pelo modelo.
- `evaluate.py` — reavalia o modelo no conjunto de teste e investiga falsos positivos e falsos negativos.

## Como executar

### 1. Pré-requisitos

- Python instalado.
- Git, caso deseje clonar o repositório.

### 2. Obter o projeto

```bash
git clone <URL_DO_REPOSITORIO>
cd industrial-predictive-maintenance
```

Substitua `<URL_DO_REPOSITORIO>` pela URL do repositório.

### 3. Criar e ativar o ambiente virtual

No Windows, usando PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

### 5. Preparar o dataset

Baixe o dataset pela página oficial do [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

Coloque o arquivo CSV no seguinte caminho, mantendo o nome esperado pelo código:

```text
data/ai4i2020.csv
```

### 6. Explorar os dados

```bash
python src/explore.py
```

### 7. Treinar o modelo

```bash
python src/train.py
```

O treinamento gera o arquivo `models/failure_model.joblib`, utilizado pelos scripts de previsão e avaliação.

### 8. Avaliar o modelo

```bash
python src/evaluate.py
```

O script apresenta as métricas de classificação, a matriz de confusão e exemplos de falsos positivos e falsos negativos.

### 9. Testar uma classificação

```bash
python src/predict.py
```

Informe o tipo da máquina (`L`, `M` ou `H`) e os valores solicitados no terminal.

O modelo retorna a classe prevista e uma pontuação estimada de falha.

## Resultados experimentais

Na avaliação realizada com a divisão de teste de 20%, o modelo apresentou os seguintes resultados:

| Métrica | Resultado |
|---|---:|
| Registros de teste | 2.000 |
| Precisão da classe de falha | 0,71 |
| Recall da classe de falha | 0,66 |
| F1-score da classe de falha | 0,69 |
| Average Precision | 0,7615 |
| Acurácia geral | 98% |

### Matriz de confusão

| | Previsto: sem falha | Previsto: com falha |
|---|---:|---:|
| Real: sem falha | 1.914 | 18 |
| Real: com falha | 23 | 45 |

O modelo identificou 45 dos 68 registros de falha presentes no conjunto de teste e deixou de identificar 23. Também classificou incorretamente 18 registros sem falha como casos de falha.

**Interpretação:** a acurácia de 98% não deve ser analisada isoladamente, pois a classe sem falha é predominante. O recall de aproximadamente 66% indica que o modelo identificou cerca de dois terços dos registros associados a falhas no conjunto avaliado.

Os resultados são específicos dessa divisão dos dados e não representam uma garantia de desempenho em outros conjuntos ou em condições industriais reais.

## Limitações e próximos passos

- O dataset é sintético e não substitui dados reais de sensores e equipamentos industriais.
- O experimento classifica registros associados a falhas; não valida a previsão temporal de falhas futuras.
- As pontuações produzidas por `predict_proba` não devem ser interpretadas como probabilidades de risco real calibradas.
- Falsos negativos e falsos positivos exigem análise, pois seus impactos operacionais podem ser diferentes.
- Não há integração com sensores, API, banco de dados industrial ou sistema de monitoramento.

Possíveis evoluções incluem investigar a importância das variáveis, comparar modelos, ajustar o limiar de classificação e avaliar o desempenho em dados temporais ou reais, quando disponíveis.

## Licença e fonte dos dados

Dataset: [AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

Consulte a página oficial para informações sobre autoria, licença e condições de uso dos dados.

---

Desenvolvido como projeto independente de estudo em Python e Machine Learning.
