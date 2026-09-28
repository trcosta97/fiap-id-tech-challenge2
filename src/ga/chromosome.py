"""
Definição do cromossomo e do espaço de busca de hiperparâmetros
da Regressão Logística.

Cada indivíduo é representado por um dicionário de hiperparâmetros.
O espaço de busca foi definido com base nas opções do sklearn e nas
características do dataset Breast Cancer (ver docs/decisoes.md).
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Espaço de busca dos hiperparâmetros
# ---------------------------------------------------------------------------
# Formato: nome_do_param → lista de valores válidos
# Os valores foram escolhidos para cobrir um intervalo clinicamente relevante
# sem tornar o espaço de busca inviável para o AG.

SEARCH_SPACE: dict[str, list[Any]] = {
    # Inverso da regularização: valores menores = regularização mais forte
    "C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
    # Tipo de penalidade
    "penalty": ["l2", "l1"],
    # Solver (deve ser compatível com a penalidade escolhida)
    "solver": ["lbfgs", "saga", "liblinear"],
    # Número máximo de iterações
    "max_iter": [200, 500, 1000, 2000],
    # Tolerância de convergência
    "tol": [1e-5, 1e-4, 1e-3],
}

# Solvers compatíveis por penalidade (restrição do sklearn)
_SOLVER_COMPAT: dict[str, list[str]] = {
    "l2": ["lbfgs", "saga", "liblinear"],
    "l1": ["saga", "liblinear"],
}


@dataclass
class Chromosome:
    """
    Representa um indivíduo na população do AG.

    Atributos
    ----------
    genes : dict
        Dicionário hiperparâmetro → valor.
    fitness : float
        F1-score (classe Dead) avaliado via validação cruzada.
        Inicializado como -1 (não avaliado).
    """

    genes: dict[str, Any] = field(default_factory=dict)
    fitness: float = -1.0

    def is_evaluated(self) -> bool:
        """Retorna True se o fitness já foi calculado."""
        return self.fitness >= 0.0

    def copy(self) -> "Chromosome":
        """Retorna uma cópia profunda do cromossomo."""
        return Chromosome(genes=deepcopy(self.genes), fitness=self.fitness)

    def to_sklearn_params(self) -> dict[str, Any]:
        """
        Retorna os genes no formato esperado pelo sklearn
        (chaves prefixadas com 'clf__' para uso no Pipeline).
        """
        return {f"clf__{k}": v for k, v in self.genes.items()}

    def __repr__(self) -> str:
        return f"Chromosome(fitness={self.fitness:.4f}, genes={self.genes})"


def _fix_solver(genes: dict[str, Any]) -> dict[str, Any]:
    """
    Garante que o solver seja compatível com a penalidade escolhida.
    Se não for, sorteia um solver compatível.
    """
    penalty = genes.get("penalty", "l2")
    solver = genes.get("solver", "lbfgs")
    compatible = _SOLVER_COMPAT[penalty]
    if solver not in compatible:
        genes["solver"] = random.choice(compatible)
    return genes


def random_chromosome(rng: np.random.Generator | None = None) -> Chromosome:
    """
    Cria um cromossomo com genes aleatórios dentro do espaço de busca.

    Parâmetros
    ----------
    rng : np.random.Generator, opcional
        Gerador de números aleatórios para reprodutibilidade.

    Retorna
    -------
    Chromosome
        Cromossomo inicializado aleatoriamente.
    """
    if rng is not None:
        # Usa índices via rng e escolhe da lista
        genes = {
            param: values[int(rng.integers(0, len(values)))]
            for param, values in SEARCH_SPACE.items()
        }
    else:
        genes = {
            param: random.choice(values)
            for param, values in SEARCH_SPACE.items()
        }

    _fix_solver(genes)
    return Chromosome(genes=genes)


def initialize_population(
    pop_size: int,
    seed: int = 42,
) -> list[Chromosome]:
    """
    Gera a população inicial com ``pop_size`` cromossomos aleatórios.

    Parâmetros
    ----------
    pop_size : int
        Tamanho da população.
    seed : int
        Seed para reprodutibilidade.

    Retorna
    -------
    list[Chromosome]
        Lista de cromossomos não avaliados.
    """
    rng = np.random.default_rng(seed)
    population = [random_chromosome(rng) for _ in range(pop_size)]
    return population
