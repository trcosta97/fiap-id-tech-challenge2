# FIAP PosTech – IA para Devs – Tech Challenge Fase 2

## Visão Geral

Este repositório contém a solução do **Tech Challenge da Fase 2** do curso PosTech – IA para Devs (IADT) da FIAP.

O cenário é um **hospital universitário** que busca resolver dois desafios com **Algoritmos Genéticos (AG)** e **Modelos de Linguagem (LLMs)**:

- **Projeto 1:** Otimização de modelos de diagnóstico via AG + interpretação dos resultados por LLM.
- **Projeto 2:** Otimização de rotas de entrega de medicamentos/insumos (TSP/VRP via AG) + geração de instruções e relatórios por LLM.

> ⚠️ Este repositório implementa o **Projeto [1 ou 2 – preencher após definição]**.

---

## Estrutura do Projeto

```
.
├── README.md
├── pyproject.toml            # Gerenciamento de dependências (Poetry)
├── .env.example              # Variáveis de ambiente necessárias (sem valores reais)
├── src/
│   ├── ga/                   # Algoritmo genético (representação, operadores, fitness)
│   ├── llm/                  # Integração com LLM, prompts, avaliação
│   ├── models/               # Modelos do Módulo 1 (Projeto 1)
│   ├── routing/              # TSP/VRP, restrições, visualização de mapa (Projeto 2)
│   ├── monitoring/           # Logging e métricas
│   └── api/                  # API (opcional)
├── tests/                    # Testes automatizados
├── notebooks/                # Notebooks de demonstração
├── experiments/              # Resultados dos experimentos (csv, json, gráficos)
├── docs/
│   ├── arquitetura.md        # Diagrama e descrição da arquitetura
│   ├── decisoes.md           # Registro das decisões de implementação
│   └── relatorio_tecnico.md  # Rascunho do relatório técnico final
└── infra/                    # Infraestrutura como Código (IaC) – somente se nuvem
```

---

## Como Executar

### Pré-requisitos

- Python 3.11+
- [Poetry](https://python-poetry.org/) (recomendado)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/trcosta97/fiap-id-tech-challenge2.git
cd fiap-id-tech-challenge2

# Instale as dependências
poetry install

# Copie e configure as variáveis de ambiente
cp .env.example .env
# Edite .env com suas chaves de API e configurações locais
```

### Executando os testes

```bash
poetry run pytest tests/
```

---

## Variáveis de Ambiente

Veja `.env.example` para a lista completa de variáveis necessárias.  
**Nunca commite o arquivo `.env` com valores reais.**

---

## Documentação

- [Arquitetura](docs/arquitetura.md)
- [Decisões de Implementação](docs/decisoes.md)
- [Relatório Técnico](docs/relatorio_tecnico.md)

---

## Integrantes do Grupo

| Nome | RM |
|------|----|
| (preencher) | (preencher) |

---

## Licença

Uso acadêmico – FIAP PosTech 2026.
