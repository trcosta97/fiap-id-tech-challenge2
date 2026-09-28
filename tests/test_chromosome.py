"""
Testes para src/ga/chromosome.py
"""

import pytest
from src.ga.chromosome import (
    SEARCH_SPACE,
    Chromosome,
    _fix_solver,
    initialize_population,
    random_chromosome,
)


class TestChromosome:
    def test_random_chromosome_has_all_genes(self):
        c = random_chromosome()
        assert set(c.genes.keys()) == set(SEARCH_SPACE.keys())

    def test_random_chromosome_genes_in_search_space(self):
        c = random_chromosome()
        for param, value in c.genes.items():
            assert value in SEARCH_SPACE[param], (
                f"Valor '{value}' inválido para '{param}'"
            )

    def test_solver_compatibility_l2(self):
        genes = {"penalty": "l2", "solver": "saga"}
        _fix_solver(genes)
        assert genes["solver"] in ["lbfgs", "saga", "liblinear"]

    def test_solver_compatibility_l1_lbfgs_is_fixed(self):
        """lbfgs não suporta l1 – deve ser corrigido."""
        genes = {"penalty": "l1", "solver": "lbfgs"}
        _fix_solver(genes)
        assert genes["solver"] in ["saga", "liblinear"]

    def test_chromosome_copy_is_independent(self):
        c = random_chromosome()
        c2 = c.copy()
        c2.genes["C"] = 9999.0
        assert c.genes["C"] != 9999.0

    def test_is_evaluated_false_by_default(self):
        c = random_chromosome()
        assert not c.is_evaluated()

    def test_is_evaluated_true_after_assignment(self):
        c = random_chromosome()
        c.fitness = 0.42
        assert c.is_evaluated()

    def test_initialize_population_size(self):
        pop = initialize_population(pop_size=10, seed=42)
        assert len(pop) == 10

    def test_initialize_population_valid_genes(self):
        pop = initialize_population(pop_size=5, seed=7)
        for ind in pop:
            for param, value in ind.genes.items():
                assert value in SEARCH_SPACE[param]

    def test_to_sklearn_params_prefix(self):
        c = random_chromosome()
        params = c.to_sklearn_params()
        for key in params:
            assert key.startswith("clf__")
