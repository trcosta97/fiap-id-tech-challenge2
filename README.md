# FIAP PosTech – IA para Devs – Tech Challenge Fase 2

## Visão Geral

**Projeto 1 – Otimização de Regressão Logística via Algoritmo Genético + LLM**

Este repositório implementa a solução do **Tech Challenge da Fase 2** do curso PosTech – IA para Devs (IADT) da FIAP.

O cenário é um **hospital universitário** que busca otimizar modelos preditivos de diagnóstico oncológico combinando:

- **Algoritmo Genético (AG):** otimização dos hiperparâmetros da Regressão Logística sobre o dataset Breast Cancer
- **LLM local (Ollama qwen3:8b):** geração de laudos interpretativos em linguagem natural

> ⚠️ **Aviso:** As saídas geradas pela LLM são de **apoio à decisão clínica** e **não substituem** o julgamento de profissional de saúde habilitado.

---

## Resultados dos Experimentos

### Módulo 1 – Baseline (3 classificadores)

| Modelo | F1 (Dead) | ROC-AUC | Recall (Dead) | Accuracy |
|--------|-----------|---------|--------------|----------|
| **Regressão Logística** | **0.3773** | **0.7185** | **0.5935** | 0.7006 |
| Random Forest | 0.2857 | 0.6966 | 0.2276 | 0.8261 |
| XGBoost | 0.3456 | 0.6821 | 0.3821 | 0.7789 |

**A Regressão Logística foi escolhida para otimização** por ter o melhor F1 Dead e ROC-AUC.

### Módulo 2 – Otimização via AG

| Experimento | Pop | Mut | Torneio | F1 (Dead) | ROC-AUC | Recall (Dead) | Melhor configuração |
|-------------|-----|-----|---------|-----------|---------|--------------|---------------------|
| Exp1 (base) | 20 | 0.10 | 3 | 0.3711 | 0.7204 | 0.5854 | C=0.1, l1, saga |
| Exp2 (alta mut) | 20 | 0.30 | 3 | 0.3673 | 0.7199 | 0.5854 | C=0.1, l2, lbfgs |
| Exp3 (pop grande) | 40 | 0.10 | 5 | 0.3711 | 0.7204 | 0.5854 | C=0.1, l1, saga |

**Destaques:** O AG melhorou o ROC-AUC em todos os experimentos (0.7185 → 0.7204) e o CV F1 Médio subiu de 0.4108 para 0.4156–0.4160. Todos os experimentos convergiram para `C=0.1` (regularização forte).

---

## Estrutura do Projeto

```
fiap-id-tech-challenge2/
├── README.md
├── pyproject.toml                # Metadados e dependências (compatível com Poetry)
├── requirements.txt              # Dependências para instalação via pip
├── .env.example                  # Variáveis de ambiente necessárias (sem valores reais)
├── src/
│   ├── models/
│   │   ├── dataset.py            # Carregamento e pré-processamento do CSV
│   │   └── train.py              # Treinamento dos 3 modelos baseline (LR, RF, XGBoost)
│   ├── ga/
│   │   ├── chromosome.py         # Cromossomo e espaço de busca de hiperparâmetros
│   │   ├── operators.py          # Seleção torneio, crossover uniforme, mutação, elitismo
│   │   ├── fitness.py            # Função fitness: F1 Dead via 5-fold StratCV
│   │   └── engine.py             # Loop evolutivo com parada antecipada
│   ├── llm/
│   │   ├── client.py             # Integração Ollama qwen3:8b + modo mock
│   │   └── prompts/
│   │       └── diagnostico_pt.txt # Prompt versionado em PT-BR
│   └── monitoring/
│       └── logger.py             # Logging estruturado com Loguru
├── experiments/
│   ├── run_experiments.py        # Script dos 3 experimentos
│   ├── baseline_metrics.json     # Métricas do Módulo 1
│   ├── exp1_baseline_ag_results.json
│   ├── exp2_alta_mutacao_results.json
│   ├── exp3_populacao_grande_results.json
│   ├── comparativo_final.json    # Tabela comparativa completa
│   └── *.png                     # Curvas de convergência do AG
├── tests/                        # 34 testes automatizados
├── data/
│   └── Breast_Cancer.csv         # Dataset (4024 amostras, 16 colunas)
├── docs/
│   ├── arquitetura.md            # Diagrama Mermaid e descrição dos componentes
│   ├── decisoes.md               # 10 decisões de arquitetura registradas
│   └── relatorio_tecnico.md      # Relatório técnico completo
├── logs/                         # Logs gerados em runtime (gitignored)
├── notebooks/                    # Notebooks de demonstração
└── infra/                        # Infraestrutura (não utilizada – execução local)
```

---

## Como Executar

### Pré-requisitos

- Python 3.11+
- [Ollama](https://ollama.com/) instalado localmente (para execução real da LLM)
- Git

### 1. Clone o repositório

```bash
git clone https://github.com/trcosta97/fiap-id-tech-challenge2.git
cd fiap-id-tech-challenge2
```

### 2. Crie e ative um ambiente virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências (pip)

```bash
pip install -r requirements.txt
```

> **Nota:** O projeto usa pip direto. O arquivo `pyproject.toml` existe para metadados e compatibilidade futura com Poetry, mas **não é necessário instalar Poetry** para executar o projeto.

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
# Edite .env conforme necessário
```

Principais variáveis (ver `.env.example`):

```env
# Ativar modo mock da LLM (não requer Ollama em execução)
LLM_MOCK=false

# URL do Ollama (padrão)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

### 5. Instale o modelo Ollama (para LLM real)

```bash
# Instale o Ollama: https://ollama.com/download
ollama pull qwen3:8b
ollama serve  # em outro terminal
```

Para executar **sem** Ollama (modo mock), use `LLM_MOCK=true` no `.env`.

### 6. Execute os experimentos

```bash
# Executar todos os 3 experimentos
python experiments/run_experiments.py
```

Os resultados serão salvos em `experiments/` (JSONs e gráficos PNG).

---

## Executando os Testes

```bash
# Todos os testes (modo mock ativado automaticamente)
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=term-missing
```

> Os 34 testes passam sem necessidade do Ollama em execução (modo mock).

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `LLM_MOCK` | `false` | Ativa modo mock da LLM (sem Ollama) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do servidor Ollama |
| `OLLAMA_MODEL` | `qwen3:8b` | Modelo a ser utilizado |
| `LOG_LEVEL` | `INFO` | Nível de logging |

**Nunca commite o arquivo `.env` com valores reais.**

---

## Documentação

| Documento | Descrição |
|-----------|-----------|
| [docs/arquitetura.md](docs/arquitetura.md) | Diagrama Mermaid completo, descrição de cada componente, decisões de infra |
| [docs/decisoes.md](docs/decisoes.md) | 10 decisões de arquitetura registradas no formato Contexto/Opções/Decisão/Justificativa/Consequências |
| [docs/relatorio_tecnico.md](docs/relatorio_tecnico.md) | Relatório técnico completo com todos os resultados |

---

## Aviso sobre a LLM

> ⚠️ **IMPORTANTE:** As respostas geradas pelo modelo de linguagem (Ollama qwen3:8b) são exclusivamente instrumentos de **apoio à decisão** do profissional de saúde. Elas **não constituem diagnóstico médico**, **não substituem avaliação clínica** e **não devem ser utilizadas como única base para decisões terapêuticas**. O sistema é destinado a profissionais de saúde habilitados.

---

## Integrantes do Grupo

| Nome | RM |
|------|----|
| Thiago Costa | (RM a preencher) |

---

## Licença

Uso acadêmico – FIAP PosTech 2026.
