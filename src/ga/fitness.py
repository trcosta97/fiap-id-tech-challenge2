"""
Função de fitness para o AG de otimização da Regressão Logística.

O fitness é o F1-score da classe Dead (positiva) avaliado via
validação cruzada estratificada de 5 folds no conjunto de treino.

Justificativa: o F1-score da classe Dead foi a métrica de decisão
do Módulo 1, pois minimizar falsos negativos (predizer Alive para
quem vai morrer) é a prioridade clínica.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.ga.chromosome import Chromosome
from src.models.dataset import build_preprocessor
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate_fitness(
    individual: Chromosome,
    X_train: Any,
    y_train: np.ndarray,
) -> float:
    """
    Calcula o fitness de um cromossomo treinando a Regressão Logística
    com os hiperparâmetros codificados nos genes.

    Parâmetros
    ----------
    individual : Chromosome
        Indivíduo a ser avaliado.
    X_train : DataFrame
        Features de treino (não transformadas; o preprocessor é parte do pipeline).
    y_train : ndarray
        Rótulos de treino.

    Retorna
    -------
    float
        F1-score médio em 5-fold CV (classe Dead). Retorna 0.0 em caso de erro.
    """
    if individual.is_evaluated():
        return individual.fitness

    try:
        pipeline = Pipeline([
            ("pre", build_preprocessor()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                random_state=42,
                **individual.genes,
            )),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train, cv=_CV, scoring="f1", n_jobs=-1
        )
        fitness = float(scores.mean())

    except Exception as exc:
        logger.warning(f"Erro ao avaliar indivíduo {individual.genes}: {exc}")
        fitness = 0.0

    individual.fitness = fitness
    return fitness


def evaluate_population(
    population: list[Chromosome],
    X_train: Any,
    y_train: np.ndarray,
) -> list[Chromosome]:
    """
    Avalia todos os indivíduos não avaliados da população (in-place).

    Parâmetros
    ----------
    population : list[Chromosome]
        Lista de cromossomos.
    X_train, y_train :
        Dados de treino.

    Retorna
    -------
    list[Chromosome]
        A mesma lista com fitness preenchido.
    """
    unevaluated = [ind for ind in population if not ind.is_evaluated()]
    logger.debug(f"Avaliando {len(unevaluated)} indivíduos não avaliados...")

    for ind in unevaluated:
        evaluate_fitness(ind, X_train, y_train)

    return population
