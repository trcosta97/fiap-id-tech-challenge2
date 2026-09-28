"""
Script principal para rodar os 3 experimentos do AG e gerar:
  - experiments/<nome>_results.json  (métricas e histórico)
  - experiments/<nome>_convergencia.png  (gráfico de convergência)
  - experiments/comparativo_final.json   (baseline vs. otimizado)
  - experiments/comparativo_final.png    (gráfico comparativo)

Uso:
    poetry run python experiments/run_experiments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# Adiciona a raiz do projeto ao path para importações absolutas
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ga.engine import GAConfig, run_ga
from src.models.dataset import build_preprocessor, prepare_data
from src.models.train import run_baseline
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

_EXPERIMENTS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configurações dos 3 experimentos
# ---------------------------------------------------------------------------
CONFIGS = [
    GAConfig(
        experiment_name="exp1_baseline_ag",
        pop_size=20,
        n_generations=30,
        mutation_rate=0.10,
        crossover_rate=0.80,
        tournament_size=3,
        elite_size=2,
        seed=42,
    ),
    GAConfig(
        experiment_name="exp2_alta_mutacao",
        pop_size=20,
        n_generations=30,
        mutation_rate=0.30,   # mutação alta → mais exploração
        crossover_rate=0.80,
        tournament_size=3,
        elite_size=2,
        seed=42,
    ),
    GAConfig(
        experiment_name="exp3_populacao_grande",
        pop_size=40,          # população maior → mais diversidade
        n_generations=30,
        mutation_rate=0.10,
        crossover_rate=0.80,
        tournament_size=5,    # torneio maior → pressão seletiva maior
        elite_size=4,
        seed=42,
    ),
]


def plot_convergencia(history, config: GAConfig) -> None:
    """Gera e salva o gráfico de convergência de um experimento."""
    generations = [s.generation for s in history]
    best_f = [s.best_fitness for s in history]
    mean_f = [s.mean_fitness for s in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(generations, best_f, label="Melhor fitness", linewidth=2, color="steelblue")
    ax.plot(generations, mean_f, label="Fitness médio", linewidth=1.5,
            linestyle="--", color="darkorange")
    ax.set_xlabel("Geração")
    ax.set_ylabel("F1-score (classe Dead) – 5-fold CV")
    ax.set_title(
        f"Convergência do AG – {config.experiment_name}\n"
        f"pop={config.pop_size} | mut={config.mutation_rate} | "
        f"cross={config.crossover_rate} | torneio={config.tournament_size}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = _EXPERIMENTS_DIR / f"{config.experiment_name}_convergencia.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Gráfico de convergência salvo: {out_path}")


def evaluate_optimized(
    best_genes: dict,
    X_train, y_train, X_test, y_test,
) -> dict:
    """Avalia o melhor indivíduo no conjunto de teste (métricas completas)."""
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            random_state=42,
            **best_genes,
        )),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_dead": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_dead": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_dead": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_std": round(float(cv_f1.std()), 4),
    }


def plot_comparativo(baseline_lr: dict, optimized_results: list[dict]) -> None:
    """Gera gráfico de barras comparando baseline vs. modelos otimizados."""
    metrics = ["f1_dead", "roc_auc", "recall_dead", "precision_dead"]
    labels = ["F1 (Dead)", "ROC-AUC", "Recall (Dead)", "Precision (Dead)"]

    n_groups = len(metrics)
    n_bars = 1 + len(optimized_results)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # Baseline
    baseline_vals = [baseline_lr.get(m, 0) for m in metrics]
    ax.bar(x - width * (n_bars / 2 - 0.5), baseline_vals,
           width, label="Baseline (Módulo 1)", color=colors[0], alpha=0.85)

    # Otimizados
    for i, opt in enumerate(optimized_results, start=1):
        vals = [opt["metrics"].get(m, 0) for m in metrics]
        ax.bar(x - width * (n_bars / 2 - 0.5 - i), vals,
               width, label=opt["name"], color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Comparativo: Regressão Logística Baseline vs. Otimizada via AG")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = _EXPERIMENTS_DIR / "comparativo_final.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Gráfico comparativo salvo: {out_path}")


def main() -> None:
    logger.info("Carregando dados...")
    X_train, X_test, y_train, y_test, le = prepare_data()

    # ---- 1. Baseline dos 3 modelos do Módulo 1 ----
    logger.info("Rodando baseline dos 3 modelos...")
    baseline_results, _ = run_baseline(save=True)
    baseline_lr = next(
        (r for r in baseline_results if "Logística" in r["model"]),
        baseline_results[0],
    )
    logger.info(f"Baseline LR → F1={baseline_lr['f1_dead']} | ROC-AUC={baseline_lr['roc_auc']}")

    # ---- 2. Rodar os 3 experimentos do AG ----
    optimized_results = []

    for config in CONFIGS:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Experimento: {config.experiment_name}")
        logger.info(f"{'#'*60}")

        best_ind, history = run_ga(config, X_train, y_train)

        # Gráfico de convergência
        plot_convergencia(history, config)

        # Avaliação no conjunto de teste
        test_metrics = evaluate_optimized(best_ind.genes, X_train, y_train, X_test, y_test)

        logger.info(
            f"Resultado no teste → F1={test_metrics['f1_dead']} | "
            f"ROC-AUC={test_metrics['roc_auc']} | Recall={test_metrics['recall_dead']}"
        )
        logger.info(f"Melhores hiperparâmetros: {best_ind.genes}")

        optimized_results.append({
            "name": config.experiment_name,
            "config": {
                "pop_size": config.pop_size,
                "n_generations": config.n_generations,
                "mutation_rate": config.mutation_rate,
                "crossover_rate": config.crossover_rate,
            },
            "best_genes": best_ind.genes,
            "best_cv_fitness": best_ind.fitness,
            "metrics": test_metrics,
        })

    # ---- 3. Gráfico e JSON comparativo ----
    plot_comparativo(baseline_lr, optimized_results)

    comparativo = {
        "baseline": baseline_lr,
        "optimized": optimized_results,
    }
    out_path = _EXPERIMENTS_DIR / "comparativo_final.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparativo, f, ensure_ascii=False, indent=2)
    logger.info(f"Comparativo final salvo em: {out_path}")

    # ---- 4. Resumo final ----
    logger.info("\n" + "=" * 70)
    logger.info("RESUMO FINAL")
    logger.info(f"{'Experimento':<30} {'F1':>6} {'ROC-AUC':>8} {'Recall':>7}")
    logger.info("-" * 70)
    logger.info(
        f"{'Baseline (Módulo 1)':<30} "
        f"{baseline_lr['f1_dead']:>6.4f} "
        f"{baseline_lr['roc_auc']:>8.4f} "
        f"{baseline_lr['recall_dead']:>7.4f}"
    )
    for opt in optimized_results:
        m = opt["metrics"]
        logger.info(
            f"{opt['name']:<30} {m['f1_dead']:>6.4f} {m['roc_auc']:>8.4f} {m['recall_dead']:>7.4f}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
