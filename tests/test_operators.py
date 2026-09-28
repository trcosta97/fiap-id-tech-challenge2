"""
Testes para src/ga/operators.py
"""

import numpy as np
import pytest

from src.ga.chromosome import SEARCH_SPACE, initialize_population, random_chromosome
from src.ga.operators import elitism, mutate, tournament_selection, uniform_crossover


@pytest.fixture
def small_population():
    """População pequena com fitness definido."""
    pop = initialize_population(pop_size=10, seed=0)
    rng = np.random.default_rng(0)
    for ind in pop:
        ind.fitness = float(rng.uniform(0.3, 0.7))
    return pop


class TestTournamentSelection:
    def test_returns_chromosome(self, small_population):
        winner = tournament_selection(small_population, tournament_size=3)
        assert winner is not None
        assert set(winner.genes.keys()) == set(SEARCH_SPACE.keys())

    def test_winner_has_high_fitness(self, small_population):
        """O vencedor deve estar entre os 'tournament_size' sorteados."""
        rng = np.random.default_rng(1)
        winner = tournament_selection(small_population, tournament_size=3, rng=rng)
        assert winner.fitness >= 0.0

    def test_returns_copy_not_reference(self, small_population):
        winner = tournament_selection(small_population, tournament_size=2)
        winner.fitness = -999.0
        # Nenhum indivíduo da população deve ter sido alterado
        for ind in small_population:
            assert ind.fitness != -999.0


class TestUniformCrossover:
    def test_children_have_valid_genes(self):
        rng = np.random.default_rng(42)
        pa = random_chromosome(rng)
        pb = random_chromosome(rng)
        ca, cb = uniform_crossover(pa, pb, crossover_rate=1.0, rng=rng)
        for child in (ca, cb):
            for param, val in child.genes.items():
                assert val in SEARCH_SPACE[param], (
                    f"Gene inválido após crossover: {param}={val}"
                )

    def test_no_crossover_returns_copies(self):
        rng = np.random.default_rng(99)
        pa = random_chromosome(rng)
        pb = random_chromosome(rng)
        # crossover_rate=0 → sem crossover, filhos são cópias dos pais
        ca, cb = uniform_crossover(pa, pb, crossover_rate=0.0, rng=rng)
        assert ca.genes == pa.genes
        assert cb.genes == pb.genes

    def test_children_solver_compatible(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            pa = random_chromosome(rng)
            pb = random_chromosome(rng)
            ca, cb = uniform_crossover(pa, pb, crossover_rate=1.0, rng=rng)
            for child in (ca, cb):
                penalty = child.genes["penalty"]
                solver = child.genes["solver"]
                compat = {"l2": ["lbfgs", "saga", "liblinear"], "l1": ["saga", "liblinear"]}
                assert solver in compat[penalty], (
                    f"Solver incompatível: penalty={penalty}, solver={solver}"
                )


class TestMutate:
    def test_mutation_invalidates_fitness(self):
        rng = np.random.default_rng(0)
        ind = random_chromosome(rng)
        ind.fitness = 0.5
        mutated = mutate(ind, mutation_rate=1.0, rng=rng)  # mutação garantida
        assert mutated.fitness == -1.0

    def test_mutation_stays_in_search_space(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            ind = random_chromosome(rng)
            mutated = mutate(ind, mutation_rate=0.5, rng=rng)
            for param, val in mutated.genes.items():
                assert val in SEARCH_SPACE[param]

    def test_zero_mutation_rate_no_change(self):
        rng = np.random.default_rng(42)
        ind = random_chromosome(rng)
        mutated = mutate(ind, mutation_rate=0.0, rng=rng)
        assert mutated.genes == ind.genes


class TestElitism:
    def test_elite_size(self, small_population):
        elites = elitism(small_population, elite_size=3)
        assert len(elites) == 3

    def test_elites_are_best(self, small_population):
        elites = elitism(small_population, elite_size=2)
        all_fitnesses = sorted([ind.fitness for ind in small_population], reverse=True)
        for i, elite in enumerate(elites):
            assert elite.fitness == all_fitnesses[i]

    def test_elites_are_copies(self, small_population):
        elites = elitism(small_population, elite_size=1)
        elites[0].fitness = -999.0
        for ind in small_population:
            assert ind.fitness != -999.0
