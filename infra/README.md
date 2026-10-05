# Infraestrutura – Fora de Escopo

Esta pasta está reservada para arquivos de **Infraestrutura como Código (IaC)** caso a solução seja implantada em nuvem.

## Decisão

Conforme registrado em [`docs/decisoes.md`](../docs/decisoes.md), a execução em nuvem é **opcional** no Tech Challenge Fase 2 (rende pontuação extra). Para este projeto, a decisão foi **executar localmente** pelos seguintes motivos:

1. **Privacidade dos dados:** dados clínicos não devem trafegar para provedores externos sem contrato adequado (LGPD).
2. **LLM local:** o modelo qwen3:8b roda via Ollama na máquina do desenvolvedor, sem custo de API e sem envio de dados.
3. **Reprodutibilidade:** o ambiente local com seeds fixas e `requirements.txt` garante resultados idênticos.
4. **Escopo acadêmico:** a complexidade de IaC não agrega ao objetivo pedagógico principal (AG + LLM).

## O que seria implementado em nuvem (referência futura)

Caso esta solução fosse implantada em produção, a pasta `infra/` conteria:

```
infra/
├── terraform/          # Provisionamento de infra (ex.: AWS, GCP, Azure)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── docker/
│   ├── Dockerfile      # Imagem da aplicação
│   └── docker-compose.yml
└── k8s/                # Manifestos Kubernetes (opcional)
    ├── deployment.yaml
    └── service.yaml
```

### Recursos que seriam provisionados

| Recurso | Finalidade |
|---------|-----------|
| Container (ECS/GKE/ACI) | Execução da aplicação Python |
| Instância GPU | Inferência local da LLM (qwen3:8b) |
| Object Storage (S3/GCS) | Armazenamento do dataset e resultados |
| Auto Scaling Group | Escalabilidade automática conforme carga |
| CloudWatch/Stackdriver | Monitoramento e alertas |

> **Nota:** A execução local via `python experiments/run_experiments.py` cobre todos os requisitos obrigatórios do Tech Challenge. A nuvem é uma extensão opcional.
