"""
Motor principal do Algoritmo Genético.

Executa o loop evolutivo e retorna o histórico de convergência,
que é salvo em experiments/ para rastreabilidade.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.ga.chromosome import Chromosome, initialize_population
from src.ga.fitness import evaluate_population
from src.ga.operators import elitism, mutate, tournament_selection, uniform_crossover
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
_EXPERIMENTS_DIR.mkdir(exist_ok=True)


@dataclass
class GAConfig:
    """Configuração parametrizável de um experimento do AG."""

    pop_size: int = 20
    """Tamanho da população."""

    n_generations: int = 30
    """Número máximo de gerações."""

    crossover_rate: float = 0.8
    """Probabilidade de crossover entre dois pais."""

    mutation_rate: float = 0.1
    """Probabilidade de mutação por gene."""

    tournament_size: int = 3
    """Número de participantes no torneio de seleção."""

    elite_size: int = 2
    """Número de indivíduos preservados via elitismo."""

    seed: int = 42
    """Seed para reprodutibilidade."""

    experiment_name: str = "experimento_1"
    """Nome do experimento (usado no nome dos arquivos de saída)."""


@dataclass
class GenerationStats:
    """Estatísticas de uma única geração."""

    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_genes: dict[str, Any]


def run_ga(
    config: GAConfig,
    X_train: Any,
    y_train: np.ndarray,
) -> tuple[Chromosome, list[GenerationStats]]:
    """
    Executa o loop evolutivo completo.

    Parâmetros
    ----------
    config : GAConfig
        Configuração do AG.
    X_train, y_train :
        Dados de treino para cálculo do fitness.

    Retorna
    -------
    best : Chromosome
        Melhor indivíduo encontrado.
    history : list[GenerationStats]
        Histórico geração a geração.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Iniciando AG: {config.experiment_name}")
    logger.info(
        f"pop_size={config.pop_size} | gerações={config.n_generations} | "
        f"mutation_rate={config.mutation_rate} | crossover_rate={config.crossover_rate}"
    )
    logger.info(f"{'='*60}")

    rng = np.random.default_rng(config.seed)
    population = initialize_population(config.pop_size, seed=config.seed)
    history: list[GenerationStats] = []
    t0 = time.perf_counter()

    for gen in range(1, config.n_generations + 1):
        # --- Avaliação ---
        evaluate_population(population, X_train, y_train)

        # --- Estatísticas ---
        fitnesses = [ind.fitness for ind in population]
        best = max(population, key=lambda c: c.fitness)
        stats = GenerationStats(
            generation=gen,
            best_fitness=round(max(fitnesses), 6),
            mean_fitness=round(float(np.mean(fitnesses)), 6),
            worst_fitness=round(min(fitnesses), 6),
            best_genes=dict(best.genes),
        )
        history.append(stats)

        logger.info(
            f"Gen {gen:03d}/{config.n_generations} | "
            f"Melhor={stats.best_fitness:.4f} | "
            f"Médio={stats.mean_fitness:.4f} | "
            f"Pior={stats.worst_fitness:.4f}"
        )

        # Critério de parada antecipada: sem melhora em 10 gerações
        if gen >= 10:
            recent = [h.best_fitness for h in history[-10:]]
            if max(recent) - min(recent) < 1e-5:
                logger.info(
                    f"Parada antecipada na geração {gen}: "
                    "sem melhora nas últimas 10 gerações."
                )
                break

        # --- Nova geração ---
        new_population: list[Chromosome] = elitism(population, config.elite_size)

        while len(new_population) < config.pop_size:
            parent_a = tournament_selection(population, config.tournament_size, rng)
            parent_b = tournament_selection(population, config.tournament_size, rng)
            child_a, child_b = uniform_crossover(
                parent_a, parent_b, config.crossover_rate, rng
            )
            child_a = mutate(child_a, config.mutation_rate, rng)
            child_b = mutate(child_b, config.mutation_rate, rng)
            new_population.append(child_a)
            if len(new_population) < config.pop_size:
                new_population.append(child_b)

        population = new_population

    elapsed = time.perf_counter() - t0
    final_best = max(population, key=lambda c: c.fitness) if history else best
    evaluate_population([final_best], X_train, y_train)

    logger.info(f"AG concluído em {elapsed:.1f}s")
    logger.info(f"Melhor indivíduo: {final_best}")

    _save_experiment(config, history, final_best)
    return final_best, history


def _save_experiment(
    config: GAConfig,
    history: list[GenerationStats],
    best: Chromosome,
) -> None:
    """
    Persiste os resultados do experimento em JSON para rastreabilidade.
    """
    out = {
        "config": {
            "pop_size": config.pop_size,
            "n_generations": config.n_generations,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
            "tournament_size": config.tournament_size,
            "elite_size": config.elite_size,
            "seed": config.seed,
            "experiment_name": config.experiment_name,
        },
        "best": {
            "fitness": best.fitness,
            "genes": best.genes,
        },
        "history": [
            {
                "generation": s.generation,
                "best_fitness": s.best_fitness,
                "mean_fitness": s.mean_fitness,
                "worst_fitness": s.worst_fitness,
            }
            for s in history
        ],
    }

    path = _EXPERIMENTS_DIR / f"{config.experiment_name}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Resultados salvos em: {path}")
