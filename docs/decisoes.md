# Registro de Decisões de Implementação

> Este documento registra as principais decisões de arquitetura e design tomadas ao longo do projeto.  
> Cada entrada deve conter: data, contexto, opções consideradas, decisão tomada e justificativa.

---

## Formato de Registro

```
### [DATA] – [TÍTULO DA DECISÃO]

**Contexto:** Descreva o problema ou situação que motivou a decisão.

**Opções consideradas:**
- Opção A: ...
- Opção B: ...

**Decisão:** Qual opção foi escolhida.

**Justificativa:** Por que esta opção foi a melhor para o contexto.

**Consequências:** Impactos esperados (positivos e negativos).
```

---

## Decisões

### [2026-09-28] – Escolha do projeto

**Contexto:** O Tech Challenge oferece dois projetos. É necessário escolher um.

**Opções consideradas:**
- Projeto 1: Otimização de modelos de diagnóstico via AG + LLM.
- Projeto 2: Otimização de rotas de entrega (TSP/VRP) via AG + LLM.

**Decisão:** A definir.

**Justificativa:** A definir após alinhamento com o grupo.

---

### [2026-09-28] – Estrutura inicial do repositório

**Contexto:** Necessidade de organizar o projeto de forma clara e reprodutível desde o início.

**Opções consideradas:**
- Estrutura mínima (só `src/` e `tests/`).
- Estrutura completa conforme sugerida no enunciado.

**Decisão:** Estrutura completa com `src/`, `tests/`, `notebooks/`, `experiments/`, `docs/`, `infra/`.

**Justificativa:** Alinhada com o enunciado; facilita a rastreabilidade dos experimentos e a elaboração do relatório técnico.

**Consequências:** Repositório mais organizado desde o início; requer disciplina para manter cada artefato no lugar correto.
