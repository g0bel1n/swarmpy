import itertools
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from .aco_step import ACO_Step


class BasePheromonesUpdater(ABC):
    def __init__(
        self,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[list[float]] = None,
    ) -> None:
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.bounds = bounds

        if bounds is not None:
            self.bounds.sort()  # type: ignore
            self.bounded = True
        else : self.bounded = False

    @abstractmethod
    def update(self, G: dict[str, np.ndarray], solutions: list[list]):
        pass

    def evaporate(self, G: dict[str, np.ndarray]):
        G["e"] *= 1 - self.evaporation_rate
        return G

    def run(self, G: dict[str, np.ndarray], solutions: list[list]):
        G  = self.evaporate(G=G)
        G  = self.update(G=G, solutions=solutions)

        if self.bounded:
            G["e"][G["e"] > self.bounds[1]] = self.bounds[1]  # type: ignore
            G["e"][G["e"] < self.bounds[0]] = self.bounds[0]  # type: ignore
        
        return {'G': G}


class ProportionnalPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def update(self, G: dict[str, np.ndarray], solutions: list[list]):

        for solution, cost in solutions:
            for i, j in itertools.pairwise(solution):
                G["e"][i, j] += self.Q / cost
                G["e"][j, i] = G["e"][i, j]
        return G


class BestSoFarPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def __init__(
        self,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[list[float]] = None,
        k: int = 5,
    ) -> None:
        super().__init__(evaporation_rate, Q, bounds)
        self.bestSoFar = []
        self.k = k

    def update(self, G: dict[str, np.ndarray], solutions: list[list]):
        if len(self.bestSoFar) == 0:
            self.bestSoFar = solutions[: self.k]
        else:
            self.bestSoFar = self.bestSoFar + solutions[: self.k]
            self.bestSoFar.sort(key=lambda x: x[1])
            self.bestSoFar = self.bestSoFar[: self.k]

        for solution, cost in self.bestSoFar:
            for i, j in itertools.pairwise(solution):
                G["e"][i, j] += self.Q / cost
                G["e"][j, i] = G["e"][i, j]
        return G

class BestTourPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def __init__(
        self,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[list[float]] = None,
        k: int = 5,
    ) -> None:
        super().__init__(evaporation_rate, Q, bounds)
        self.k = k

    def update(self, G: dict[str, np.ndarray], solutions: list[list]):

        for solution, cost in solutions[: self.k]:
            for i, j in itertools.pairwise(solution):
                G["e"][i, j] += self.Q / cost
                G["e"][j, i] = G["e"][i, j]

        return G