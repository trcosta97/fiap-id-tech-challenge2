"""
Testes para src/ga/fitness.py
"""

import numpy as np
import pytest

from src.ga.chromosome import random_chromosome
from src.ga.fitness import evaluate_fitness, evaluate_population


class TestEvaluateFitness:
    def test_returns_float_between_0_and_1(self, small_dataset):
        X, y = small_dataset
        ind = random_chromosome()
        fitness = evaluate_fitness(ind, X, y)
        assert 0.0 <= fitness <= 1.0, f"Fitness fora do intervalo: {fitness}"

    def test_fitness_cached(self, small_dataset):
        """Segunda chamada não deve recalcular (fitness já definido)."""
        X, y = small_dataset
        ind = random_chromosome()
        f1 = evaluate_fitness(ind, X, y)
        # Altera manualmente para ver se a função respeita o cache
        ind.fitness = 0.9999
        f2 = evaluate_fitness(ind, X, y)
        assert f2 == 0.9999

    def test_fitness_is_set_on_individual(self, small_dataset):
        X, y = small_dataset
        ind = random_chromosome()
        assert not ind.is_evaluated()
        evaluate_fitness(ind, X, y)
        assert ind.is_evaluated()


class TestEvaluatePopulation:
    def test_all_individuals_evaluated(self, small_dataset):
        X, y = small_dataset
        pop = [random_chromosome() for _ in range(5)]
        evaluate_population(pop, X, y)
        for ind in pop:
            assert ind.is_evaluated()

    def test_already_evaluated_not_overwritten(self, small_dataset):
        X, y = small_dataset
        pop = [random_chromosome() for _ in range(3)]
        pop[0].fitness = 0.777  # pré-avaliado
        evaluate_population(pop, X, y)
        assert pop[0].fitness == 0.777  # não deve ser sobrescrito
