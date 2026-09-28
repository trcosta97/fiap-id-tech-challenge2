"""
Operadores genéticos: seleção por torneio, crossover uniforme e mutação.

Design decisions (ver docs/decisoes.md):
- Seleção por torneio: robusto a populações pequenas, sem pressão seletiva
  excessiva nos primeiros estágios.
- Crossover uniforme: cada gene é herdado independentemente de um dos pais
  com probabilidade 0.5 – exploração ampla do espaço combinatório.
- Mutação por substituição aleatória: troca um gene por outro valor válido
  do espaço de busca, garantindo que o indivíduo mutado seja sempre válido.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from src.ga.chromosome import (
    SEARCH_SPACE,
    Chromosome,
    _fix_solver,
)


# ---------------------------------------------------------------------------
# Seleção por torneio
# ---------------------------------------------------------------------------

def tournament_selection(
    population: list[Chromosome],
    tournament_size: int = 3,
    rng: np.random.Generator | None = None,
) -> Chromosome:
    """
    Seleciona um indivíduo via torneio: escolhe ``tournament_size``
    candidatos aleatoriamente e retorna o de maior fitness.

    Parâmetros
    ----------
    population : list[Chromosome]
        População atual (todos devem ter fitness calculado).
    tournament_size : int
        Número de participantes do torneio.
    rng : np.random.Generator, opcional
        Gerador para reprodutibilidade.

    Retorna
    -------
    Chromosome
        Cópia do vencedor (não modifica a população).
    """
    if rng is not None:
        indices = rng.choice(len(population), size=tournament_size, replace=False)
        candidates = [population[i] for i in indices]
    else:
        candidates = random.sample(population, k=tournament_size)

    winner = max(candidates, key=lambda c: c.fitness)
    return winner.copy()


# ---------------------------------------------------------------------------
# Crossover uniforme
# ---------------------------------------------------------------------------

def uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    crossover_rate: float = 0.8,
    rng: np.random.Generator | None = None,
) -> tuple[Chromosome, Chromosome]:
    """
    Crossover uniforme: cada gene do filho é herdado de um dos pais com
    probabilidade 0.5. A operação só ocorre com probabilidade ``crossover_rate``;
    caso contrário, os pais são retornados como cópias.

    Parâmetros
    ----------
    parent_a, parent_b : Chromosome
        Pais selecionados.
    crossover_rate : float
        Probabilidade de aplicar o crossover (padrão: 0.8).
    rng : np.random.Generator, opcional
        Gerador para reprodutibilidade.

    Retorna
    -------
    tuple[Chromosome, Chromosome]
        Dois filhos resultantes.
    """
    _rand = rng.random() if rng is not None else random.random()

    if _rand > crossover_rate:
        return parent_a.copy(), parent_b.copy()

    genes_a: dict[str, Any] = {}
    genes_b: dict[str, Any] = {}

    for param in SEARCH_SPACE:
        if rng is not None:
            flip = rng.random()
        else:
            flip = random.random()

        if flip < 0.5:
            genes_a[param] = parent_a.genes[param]
            genes_b[param] = parent_b.genes[param]
        else:
            genes_a[param] = parent_b.genes[param]
            genes_b[param] = parent_a.genes[param]

    _fix_solver(genes_a)
    _fix_solver(genes_b)

    child_a = Chromosome(genes=genes_a)
    child_b = Chromosome(genes=genes_b)
    return child_a, child_b


# ---------------------------------------------------------------------------
# Mutação por substituição aleatória
# ---------------------------------------------------------------------------

def mutate(
    individual: Chromosome,
    mutation_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Chromosome:
    """
    Mutação: cada gene é substituído por um valor aleatório do espaço de
    busca com probabilidade ``mutation_rate``.

    Parâmetros
    ----------
    individual : Chromosome
        Indivíduo a ser mutado (não é modificado in-place).
    mutation_rate : float
        Probabilidade de mutação por gene (padrão: 0.1).
    rng : np.random.Generator, opcional
        Gerador para reprodutibilidade.

    Retorna
    -------
    Chromosome
        Novo cromossomo (possivelmente mutado).
    """
    mutated = individual.copy()
    mutated.fitness = -1.0  # fitness invalidado após mutação

    for param, values in SEARCH_SPACE.items():
        _rand = rng.random() if rng is not None else random.random()
        if _rand < mutation_rate:
            if rng is not None:
                mutated.genes[param] = values[int(rng.integers(0, len(values)))]
            else:
                mutated.genes[param] = random.choice(values)

    _fix_solver(mutated.genes)
    return mutated


# ---------------------------------------------------------------------------
# Elitismo
# ---------------------------------------------------------------------------

def elitism(
    population: list[Chromosome],
    elite_size: int = 1,
) -> list[Chromosome]:
    """
    Retorna os ``elite_size`` melhores indivíduos da população (cópias),
    para preservação na próxima geração.

    Parâmetros
    ----------
    population : list[Chromosome]
        População avaliada.
    elite_size : int
        Número de elites (padrão: 1).

    Retorna
    -------
    list[Chromosome]
        Lista com as melhores cópias.
    """
    sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
    return [ind.copy() for ind in sorted_pop[:elite_size]]
