import itertools
from threading import Thread

import numpy as np

from algorithms.AS.as_ant import AS_Ant

from algorithms.BaseACO import BaseACO


def add_solution(solutions, params, G, start):
    ant = AS_Ant(params=params, G=G)
    solutions.append(ant.build_get(start=start))


class AS_Algo(BaseACO):
    def __init__(
        self,
        positions: np.ndarray,
        iter_max: int = 100,
        rho: float = 0.6,
        Q: float = 100,
    ):
        super().__init__(positions, iter_max)
        self.rho = rho
        self.Q = Q

    def _BaseACO__construct_solutions(self):
        solutions = []
        params = {"alpha": 1.0, "beta": 2.0}
        threads = [
            Thread(target=add_solution, args=(solutions, params, self.G, i))
            for i in range(self.batchSize)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        solutions.sort(key=lambda x: x[1])
        self.solutions = solutions

    def _BaseACO__update_pheromones(self):

        self.G["e"] *= 1 - self.rho  # Evaporation

        for solution, cost in self.solutions:  # Reinforcement
            for i, j in itertools.pairwise(solution):
                self.G["e"][i, j] += self.Q / cost
                self.G["e"][j, i] = self.G["e"][i, j]
