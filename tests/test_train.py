"""
Testes para src/models/train.py

Cobrem a construção dos pipelines baseline e a função run_baseline,
usando o dataset sintético do conftest quando o CSV real não está disponível.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.dataset import (
    CATEGORICAL_COLS,
    DEFAULT_CSV,
    NUMERIC_COLS,
    build_preprocessor,
)
from src.models.train import build_baseline_pipelines, run_baseline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_split():
    """
    Split treino/teste sintético pequeno.
    Não depende do CSV real – os testes de estrutura rodam sempre.
    """
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 80

    X_train = pd.DataFrame(
        np.hstack([
            rng.standard_normal((n_train, len(NUMERIC_COLS))),
            rng.integers(0, 3, size=(n_train, len(CATEGORICAL_COLS))).astype(float),
        ]),
        columns=NUMERIC_COLS + CATEGORICAL_COLS,
    )
    X_test = pd.DataFrame(
        np.hstack([
            rng.standard_normal((n_test, len(NUMERIC_COLS))),
            rng.integers(0, 3, size=(n_test, len(CATEGORICAL_COLS))).astype(float),
        ]),
        columns=NUMERIC_COLS + CATEGORICAL_COLS,
    )
    # ~15% positivos, mimetizando o desbalanceamento real
    y_train = rng.choice([0, 1], size=n_train, p=[0.85, 0.15])
    y_test = rng.choice([0, 1], size=n_test, p=[0.85, 0.15])

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Testes de build_baseline_pipelines
# ---------------------------------------------------------------------------

class TestBuildBaselinePipelines:
    def test_retorna_tres_pipelines(self, synthetic_split):
        X_train, _, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        assert len(pipelines) == 3

    def test_chaves_corretas(self, synthetic_split):
        X_train, _, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        assert "Regressão Logística" in pipelines
        assert "Random Forest" in pipelines
        assert "XGBoost" in pipelines

    def test_todos_sao_pipelines_sklearn(self, synthetic_split):
        X_train, _, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        for name, pipe in pipelines.items():
            assert isinstance(pipe, Pipeline), f"{name} não é um Pipeline sklearn"

    def test_pipelines_tem_dois_passos(self, synthetic_split):
        """Cada pipeline deve ter exatamente 'pre' e 'clf'."""
        X_train, _, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        for name, pipe in pipelines.items():
            step_names = [s[0] for s in pipe.steps]
            assert "pre" in step_names, f"Step 'pre' ausente em {name}"
            assert "clf" in step_names, f"Step 'clf' ausente em {name}"

    def test_pipelines_nao_estao_treinados(self, synthetic_split):
        """Os pipelines devem ser retornados sem ajuste (não fitted)."""
        X_train, _, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        for name, pipe in pipelines.items():
            with pytest.raises(Exception):
                # Qualquer pipeline não treinado deve falhar ao predict
                pipe.predict(X_train.head(5))

    def test_lr_tem_class_weight_balanced(self, synthetic_split):
        X_train, _, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        clf = pipelines["Regressão Logística"].named_steps["clf"]
        assert clf.class_weight == "balanced"

    def test_rf_tem_class_weight(self, synthetic_split):
        X_train, _, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        clf = pipelines["Random Forest"].named_steps["clf"]
        assert clf.class_weight is not None


# ---------------------------------------------------------------------------
# Testes de treinamento dos pipelines individuais
# ---------------------------------------------------------------------------

class TestPipelineTraining:
    def test_lr_pode_ser_treinado_e_prediz(self, synthetic_split):
        X_train, X_test, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        pipe = pipelines["Regressão Logística"]
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(preds).issubset({0, 1})

    def test_lr_predict_proba_shape(self, synthetic_split):
        X_train, X_test, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        pipe = pipelines["Regressão Logística"]
        pipe.fit(X_train, y_train)
        probas = pipe.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)
        # Probabilidades devem somar ~1 por amostra
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_rf_pode_ser_treinado(self, synthetic_split):
        X_train, X_test, y_train, _ = synthetic_split
        pipelines = build_baseline_pipelines(y_train)
        pipe = pipelines["Random Forest"]
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)


# ---------------------------------------------------------------------------
# Testes de run_baseline (integração com CSV real)
# ---------------------------------------------------------------------------

class TestRunBaseline:
    def test_run_baseline_retorna_resultados(self):
        if not DEFAULT_CSV.exists():
            pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
        results, trained = run_baseline(save=False)
        assert len(results) == 3
        assert len(trained) == 3

    def test_metricas_esperadas_presentes(self):
        if not DEFAULT_CSV.exists():
            pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
        results, _ = run_baseline(save=False)
        expected_keys = {
            "model", "accuracy", "precision_dead", "recall_dead",
            "f1_dead", "roc_auc", "cv_f1_mean", "cv_f1_std",
        }
        for r in results:
            assert expected_keys.issubset(r.keys()), (
                f"Chaves ausentes em {r.get('model')}: {expected_keys - r.keys()}"
            )

    def test_metricas_em_intervalo_valido(self):
        """Todas as métricas devem estar entre 0 e 1."""
        if not DEFAULT_CSV.exists():
            pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
        results, _ = run_baseline(save=False)
        numeric_keys = ["accuracy", "precision_dead", "recall_dead",
                        "f1_dead", "roc_auc", "cv_f1_mean", "cv_f1_std"]
        for r in results:
            for key in numeric_keys:
                val = r[key]
                assert 0.0 <= val <= 1.0, (
                    f"{r['model']}.{key} = {val} está fora do intervalo [0, 1]"
                )

    def test_lr_melhor_roc_auc(self):
        """Regressão Logística deve ter o maior ROC-AUC (conforme relatório)."""
        if not DEFAULT_CSV.exists():
            pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
        results, _ = run_baseline(save=False)
        by_model = {r["model"]: r for r in results}
        lr_auc = by_model["Regressão Logística"]["roc_auc"]
        rf_auc = by_model["Random Forest"]["roc_auc"]
        xgb_auc = by_model["XGBoost"]["roc_auc"]
        assert lr_auc >= rf_auc, f"LR ROC-AUC ({lr_auc}) < RF ({rf_auc})"
        assert lr_auc >= xgb_auc, f"LR ROC-AUC ({lr_auc}) < XGB ({xgb_auc})"

    def test_save_gera_json(self, tmp_path):
        """Com save=True o JSON deve ser gerado no diretório experiments/."""
        if not DEFAULT_CSV.exists():
            pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
        # Faz patch do diretório de saída para não sobrescrever experimentos reais
        import src.models.train as train_module
        original_dir = train_module._EXPERIMENTS_DIR
        try:
            train_module._EXPERIMENTS_DIR = tmp_path
            run_baseline(save=True)
            assert (tmp_path / "baseline_metrics.json").exists()
        finally:
            train_module._EXPERIMENTS_DIR = original_dir

    def test_save_json_conteudo_valido(self, tmp_path):
        """O JSON salvo deve ser uma lista de 3 dicionários com métricas."""
        if not DEFAULT_CSV.exists():
            pytest.skip(f"Dataset não encontrado: {DEFAULT_CSV}")
        import src.models.train as train_module
        original_dir = train_module._EXPERIMENTS_DIR
        try:
            train_module._EXPERIMENTS_DIR = tmp_path
            run_baseline(save=True)
            with open(tmp_path / "baseline_metrics.json", encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 3
            for entry in data:
                assert "model" in entry
                assert "f1_dead" in entry
        finally:
            train_module._EXPERIMENTS_DIR = original_dir
