import itertools
from threading import Thread

import numpy as np

from algorithms.ACS_ranked.acs_ant import ACS_Ant

from algorithms.BaseACO import BaseACO


def add_solution(solutions, params, G, start, q):
    ant = ACS_Ant(params=params, G=G, q=q)
    solutions.append(ant.build_get(start=start))


class ACS_Algo(BaseACO):
    def __init__(
        self,
        positions: np.ndarray,
        q: float,
        iter_max: int = 100,
        rho: float = 0.6,
        Q: float = 100,
    ):
        super().__init__(positions, iter_max)
        self.rho = rho
        self.Q = Q
        self.q = q

    def _BaseACO__construct_solutions(self):
        solutions = []
        params = {"alpha": 1.0, "beta": 3.0}
        threads = [
            Thread(target=add_solution, args=(solutions, params, self.G, i, self.q))
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

        for i, (solution, cost) in enumerate(self.solutions):  # Reinforcement
            for i, j in itertools.pairwise(solution):
                self.G["e"][i, j] += (self.Q * 1-(i/self.batchSize))/ cost
                self.G["e"][j, i] = self.G["e"][i, j]
