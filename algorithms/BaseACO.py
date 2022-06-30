from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm


def compute_distance(positions: np.ndarray) -> np.ndarray:
    distances = np.eye(positions.shape[0])
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            distances[i, j] = (
                (positions[[i, j], :][0] - positions[[i, j], :][1]) ** 2
            ).sum() ** (0.5)
            distances[j, i] = distances[i, j]
    return distances


class BaseACO(ABC):
    def __init__(self, positions: np.ndarray, iter_max: int = 100):
        self.iter_max = iter_max
        self.tau_e_0 = 1
        self.solutions = []
        self.positions = positions
        self.G: dict[str, np.ndarray]
        self.batchSize = len(positions)
        self.costs = []

    @abstractmethod
    def __construct_solutions(self):
        pass

    @abstractmethod
    def __update_pheromones(self):
        pass

    def __daemon_actions(self):
        pass

    def fit(self):

        e_pheromones = np.ones((self.batchSize, self.batchSize)) * self.tau_e_0
        heuristics = compute_distance(self.positions)

        self.G = {"e": e_pheromones, "heuristic": heuristics}

        nb_iter = 0

        #some_stagnation_condition = True

        self.costs = []
        with tqdm(total=self.iter_max, desc="Searching", ascii="░▒█") as pbar:
            while nb_iter < self.iter_max:
                self.__construct_solutions()
                self.costs.append(self.solutions[0][1])
                self.__update_pheromones()
                self.__daemon_actions()
                nb_iter += 1
                pbar.update(1)
        return self
