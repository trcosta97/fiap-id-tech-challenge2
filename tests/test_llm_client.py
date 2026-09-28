"""
Testes para src/llm/client.py

Usam mock=True para não depender do Ollama estar rodando.
"""

import pytest
from src.llm.client import _format_dados_paciente, generate_explanation


DADOS_EXEMPLO = {
    "Age": 55,
    "Race": "White",
    "Marital Status": "Married",
    "T Stage ": "T2",
    "N Stage": "N1",
    "6th Stage": "IIA",
    "differentiate": "Moderately differentiated",
    "Grade": "2",
    "A Stage": "Regional",
    "Tumor Size": 25,
    "Estrogen Status": "Positive",
    "Progesterone Status": "Positive",
    "Regional Node Examined": 12,
    "Reginol Node Positive": 1,
}

METRICAS_EXEMPLO = {
    "f1_dead": 0.377,
    "roc_auc": 0.719,
    "recall_dead": 0.594,
}


class TestFormatDadosPaciente:
    def test_returns_string(self):
        resultado = _format_dados_paciente(DADOS_EXEMPLO)
        assert isinstance(resultado, str)

    def test_contains_translated_keys(self):
        resultado = _format_dados_paciente(DADOS_EXEMPLO)
        assert "Idade" in resultado
        assert "Tamanho do tumor" in resultado

    def test_contains_values(self):
        resultado = _format_dados_paciente(DADOS_EXEMPLO)
        assert "55" in resultado
        assert "White" in resultado


class TestGenerateExplanation:
    def test_mock_returns_string(self):
        result = generate_explanation(
            dados_paciente=DADOS_EXEMPLO,
            predicao="Alive",
            probabilidade_dead=0.28,
            model_metrics=METRICAS_EXEMPLO,
            mock=True,
        )
        assert isinstance(result, str)
        assert len(result) > 50

    def test_mock_contains_predicao(self):
        result = generate_explanation(
            dados_paciente=DADOS_EXEMPLO,
            predicao="Dead",
            probabilidade_dead=0.72,
            model_metrics=METRICAS_EXEMPLO,
            mock=True,
        )
        assert "Dead" in result

    def test_mock_contains_disclaimer(self):
        """A resposta deve mencionar que é apoio à decisão."""
        result = generate_exploration = generate_explanation(
            dados_paciente=DADOS_EXEMPLO,
            predicao="Alive",
            probabilidade_dead=0.30,
            model_metrics=METRICAS_EXEMPLO,
            mock=True,
        )
        assert "apoio" in result.lower() or "decisão" in result.lower()

    def test_mock_alive_vs_dead_differ(self):
        """Respostas para Alive e Dead devem ser diferentes."""
        r_alive = generate_explanation(
            dados_paciente=DADOS_EXEMPLO,
            predicao="Alive",
            probabilidade_dead=0.2,
            model_metrics=METRICAS_EXEMPLO,
            mock=True,
        )
        r_dead = generate_explanation(
            dados_paciente=DADOS_EXEMPLO,
            predicao="Dead",
            probabilidade_dead=0.8,
            model_metrics=METRICAS_EXEMPLO,
            mock=True,
        )
        assert r_alive != r_dead
