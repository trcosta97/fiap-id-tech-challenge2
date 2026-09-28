"""
Treinamento e avaliação dos modelos baseline (Regressão Logística,
Random Forest e XGBoost) — reprodução fiel do Módulo 1.

As métricas são salvas em experiments/baseline_metrics.json para
rastreabilidade e comparação com os modelos otimizados pelo AG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.dataset import build_preprocessor, prepare_data
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
_EXPERIMENTS_DIR.mkdir(exist_ok=True)


def _compute_metrics(
    name: str,
    pipeline: Pipeline,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Treina o pipeline e retorna um dicionário com as métricas completas.

    Parâmetros
    ----------
    name : str
        Nome amigável do modelo.
    pipeline : Pipeline
        Pipeline sklearn (pré-processamento + classificador).
    X_train, X_test : DataFrame
        Features de treino e teste.
    y_train, y_test : ndarray
        Rótulos.

    Retorna
    -------
    dict
        Dicionário com métricas e o pipeline treinado.
    """
    logger.info(f"Treinando modelo: {name}")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Validação cruzada no conjunto de treino
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")

    metrics: dict[str, Any] = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_dead": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_dead": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_dead": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_std": round(float(cv_f1.std()), 4),
    }

    logger.info(
        f"{name} → F1={metrics['f1_dead']} | ROC-AUC={metrics['roc_auc']} | "
        f"Recall={metrics['recall_dead']} | CV F1={metrics['cv_f1_mean']}±{metrics['cv_f1_std']}"
    )
    return metrics, pipeline


def build_baseline_pipelines(y_train: np.ndarray) -> dict[str, Pipeline]:
    """
    Monta os três pipelines baseline exatamente como no Módulo 1.

    Parâmetros
    ----------
    y_train : ndarray
        Rótulos de treino (necessário para calcular scale_pos_weight do XGBoost).

    Retorna
    -------
    dict[str, Pipeline]
        Dicionário nome → pipeline (não treinado).
    """
    preprocessor = build_preprocessor()

    # --- Regressão Logística (melhor modelo do Módulo 1) ---
    lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )),
    ])

    # --- Random Forest ---
    rf = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # --- XGBoost ---
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pw = neg / pos
    logger.info(f"XGBoost scale_pos_weight = {neg}/{pos} = {scale_pw:.2f}")

    xgb = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pw,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return {
        "Regressão Logística": lr,
        "Random Forest": rf,
        "XGBoost": xgb,
    }


def run_baseline(save: bool = True) -> list[dict]:
    """
    Treina e avalia os três modelos baseline; salva métricas em
    ``experiments/baseline_metrics.json``.

    Parâmetros
    ----------
    save : bool
        Se True, persiste o JSON de métricas (padrão: True).

    Retorna
    -------
    list[dict]
        Lista de dicionários com métricas de cada modelo.
    """
    X_train, X_test, y_train, y_test, le = prepare_data()
    pipelines = build_baseline_pipelines(y_train)

    results = []
    trained: dict[str, Pipeline] = {}

    for name, pipeline in pipelines.items():
        metrics, fitted_pipe = _compute_metrics(
            name, pipeline, X_train, y_train, X_test, y_test
        )
        results.append(metrics)
        trained[name] = fitted_pipe

    if save:
        out_path = _EXPERIMENTS_DIR / "baseline_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Métricas baseline salvas em: {out_path}")

    # Log comparativo final
    logger.info("=" * 60)
    logger.info("COMPARATIVO BASELINE")
    logger.info(f"{'Modelo':<25} {'F1':>6} {'ROC-AUC':>8} {'Recall':>7}")
    logger.info("-" * 60)
    for r in results:
        logger.info(
            f"{r['model']:<25} {r['f1_dead']:>6.4f} {r['roc_auc']:>8.4f} {r['recall_dead']:>7.4f}"
        )
    logger.info("=" * 60)

    return results, trained
