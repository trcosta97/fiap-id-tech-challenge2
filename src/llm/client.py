"""
Cliente de integração com Ollama para geração de explicações de diagnóstico.

Usa o modelo qwen3:8b (disponível localmente) para gerar texto em PT-BR.
Os prompts são versionados em src/llm/prompts/ e carregados dinamicamente.

Design decisions (ver docs/decisoes.md):
- Ollama local: sem custo de API, sem envio de dados de pacientes para a nuvem.
- Prompts em arquivos .txt: versionamento fácil sem alterar código.
- Mock embutido: testes não dependem do Ollama estar rodando.
"""

from __future__ import annotations

import os
from pathlib import Path
from string import Template

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Modelo padrão configurável via variável de ambiente
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))


def _load_prompt_template(template_name: str) -> Template:
    """
    Carrega um template de prompt do diretório prompts/.

    Parâmetros
    ----------
    template_name : str
        Nome do arquivo sem extensão (ex: 'diagnostico_pt').

    Retorna
    -------
    string.Template
        Template com placeholders ${...} prontos para substituição.
    """
    path = _PROMPTS_DIR / f"{template_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Template de prompt não encontrado: {path}")
    return Template(path.read_text(encoding="utf-8"))


def _format_dados_paciente(dados: dict) -> str:
    """
    Formata o dicionário de dados do paciente em texto legível.
    """
    traducoes = {
        "Age": "Idade",
        "Race": "Raça/Etnia",
        "Marital Status": "Estado Civil",
        "T Stage ": "T Stage (tumor primário)",
        "N Stage": "N Stage (linfonodos)",
        "6th Stage": "Estadiamento (6ª ed.)",
        "differentiate": "Diferenciação celular",
        "Grade": "Grau histológico",
        "A Stage": "Estágio clínico",
        "Tumor Size": "Tamanho do tumor (mm)",
        "Estrogen Status": "Status receptor estrogênio",
        "Progesterone Status": "Status receptor progesterona",
        "Regional Node Examined": "Linfonodos examinados",
        "Reginol Node Positive": "Linfonodos positivos",
    }
    linhas = []
    for chave, valor in dados.items():
        nome_pt = traducoes.get(chave, chave)
        linhas.append(f"- {nome_pt}: {valor}")
    return "\n".join(linhas)


def generate_explanation(
    dados_paciente: dict,
    predicao: str,
    probabilidade_dead: float,
    model_metrics: dict,
    model: str = DEFAULT_MODEL,
    mock: bool = False,
) -> str:
    """
    Gera uma explicação em PT-BR do diagnóstico via Ollama.

    Parâmetros
    ----------
    dados_paciente : dict
        Features do paciente (nomes originais do dataset).
    predicao : str
        'Alive' ou 'Dead'.
    probabilidade_dead : float
        Probabilidade estimada de Dead (0.0 a 1.0).
    model_metrics : dict
        Métricas do modelo (f1_dead, roc_auc, recall_dead).
    model : str
        Modelo Ollama a usar (padrão: qwen3:8b).
    mock : bool
        Se True, retorna resposta simulada sem chamar o Ollama
        (útil para testes automatizados).

    Retorna
    -------
    str
        Explicação gerada em português.
    """
    template = _load_prompt_template("diagnostico_pt")
    prompt = template.substitute(
        dados_paciente=_format_dados_paciente(dados_paciente),
        predicao=predicao,
        probabilidade_dead=probabilidade_dead,
        probabilidade_alive=1.0 - probabilidade_dead,
        f1_dead=model_metrics.get("f1_dead", "N/A"),
        roc_auc=model_metrics.get("roc_auc", "N/A"),
        recall_dead=model_metrics.get("recall_dead", "N/A"),
    )

    if mock:
        logger.info("Modo mock ativado: retornando resposta simulada.")
        return _mock_response(predicao, probabilidade_dead)

    try:
        import ollama  # importação lazy – evita dependência obrigatória em testes

        logger.info(f"Chamando Ollama (modelo: {model})...")
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": float(os.getenv("LLM_TEMPERATURE", "0.3"))},
        )
        texto = response["message"]["content"]
        logger.info(f"Resposta recebida ({len(texto)} caracteres).")
        return texto

    except ImportError:
        logger.error("Pacote 'ollama' não instalado. Use: poetry add ollama")
        raise
    except Exception as exc:
        logger.error(f"Erro ao chamar Ollama: {exc}")
        raise RuntimeError(f"Falha na integração com LLM: {exc}") from exc


def _mock_response(predicao: str, probabilidade_dead: float) -> str:
    """Resposta simulada para testes sem o Ollama rodando."""
    return (
        f"**[RESPOSTA SIMULADA – MODO MOCK]**\n\n"
        f"**1. Resultado da predição**\n"
        f"O modelo classificou o paciente como **{predicao}** com probabilidade "
        f"de óbito de {probabilidade_dead:.1%}.\n\n"
        f"**2. Fatores relevantes**\n"
        f"Em geral, os receptores hormonais (estrogênio e progesterona) e o estadiamento "
        f"clínico são os fatores mais determinantes neste modelo.\n\n"
        f"**3. Interpretação clínica**\n"
        f"Este resultado indica {'risco elevado' if predicao == 'Dead' else 'prognóstico favorável'} "
        f"com base nas características clínicas registradas.\n\n"
        f"**4. Limitações**\n"
        f"O modelo foi treinado em um dataset específico e pode não generalizar para "
        f"todas as populações. O recall da classe Dead é de aproximadamente 60%.\n\n"
        f"**5. Recomendação**\n"
        f"⚠️ Este resultado é apenas uma ferramenta de **apoio à decisão**. "
        f"A conduta clínica final deve sempre ser determinada pelo profissional de saúde habilitado."
    )
