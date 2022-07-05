import itertools
from abc import ABC, abstractmethod
from typing import Any, Optional,List

import numpy as np

from .aco_step import ACO_Step


# > This class is an abstract base class for updating pheromones
class BasePheromonesUpdater(ABC):
    def __init__(
        self,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[List[float]] = None,
    ) -> None:
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.bounds = bounds

        if bounds is not None:
            self.bounds.sort()  # type: ignore
            self.bounded = True
        else:
            self.bounded = False

    @abstractmethod
    def update(
        self, G: dict[str, np.ndarray], solutions: List[list]
    ) -> dict[str, np.ndarray]:
        pass

    def evaporate(self, G: dict[str, np.ndarray], ant_params: dict[str, Any]):
        if "mask" in ant_params:
            G["e"][ant_params["mask"]] *= 1 - self.evaporation_rate
        else:
            G["e"] *= 1 - self.evaporation_rate
        return G

    def run(self, G: dict[str, np.ndarray], solutions: List[list], ant_params: list):
        G = self.evaporate(G=G, ant_params=ant_params[0])
        G = self.update(G=G, solutions=solutions)

        if self.bounded:
            G["e"][G["e"] > self.bounds[1]] = self.bounds[1]  # type: ignore
            G["e"][G["e"] < self.bounds[0]] = self.bounds[0]  # type: ignore

        return {"G": G}


# It's a pheromones updater that updates pheromones proportionnally to the quality of the solution
class ProportionnalPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def update(self, G: dict[str, np.ndarray], solutions: List[list]):

        for solution, cost in solutions:
            for i, j in itertools.pairwise(solution):
                G["e"][i, j] += self.Q / cost
                G["e"][j, i] = G["e"][i, j]
        return G


# "This class is a pheromones updater that updates the pheromones on the k best paths found so far."
class BestSoFarPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def __init__(
        self,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[List[float]] = None,
        k: int = 5,
    ) -> None:
        super().__init__(evaporation_rate, Q, bounds)
        self.bestSoFar = []
        self.k = k

    def update(self, G: dict[str, np.ndarray], solutions: List[list]):
        """
        > For each solution in the list of solutions, add the value of Q divided by the cost of the
        solution to the edge weights of the graph

        :param G: the graph
        :type G: dict[str, np.ndarray]
        :param solutions: List[list]
        :type solutions: List[list]
        :return: The updated graph.
        """
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


# "This class is a pheromones updater that updates the pheromones on the k best paths found on the tour."


class BestTourPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def __init__(
        self,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[List[float]] = None,
        k: int = 5,
    ) -> None:
        super().__init__(evaporation_rate, Q, bounds)
        self.k = k

    def update(self, G: dict[str, np.ndarray], solutions: List[list]):
        """
        > For each solution in the first k solutions, add Q/cost to the edge between each pair of nodes
        in the solution

        :param G: the graph
        :type G: dict[str, np.ndarray]
        :param solutions: List[list]
        :type solutions: List[list]
        :return: The updated graph.
        """

        for solution, cost in solutions[: self.k]:
            for i, j in itertools.pairwise(solution):
                G["e"][i, j] += self.Q / cost
                G["e"][j, i] = G["e"][i, j]

        return G
