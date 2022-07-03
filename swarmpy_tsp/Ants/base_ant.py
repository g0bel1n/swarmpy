from typing import Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseAnt(ABC):
    def __init__(self, ant_params: dict, G: dict[str, np.ndarray]) -> None:
        self.params = ant_params
        self.G = G

    @abstractmethod
    def __choose_next_node(
        self, available_nodes: np.ndarray, chosen_node: int, proba_matrix: np.ndarray
    ) -> int:
        pass

    def __compute_proba_matrix(self) -> np.ndarray:

        return (self.G["e"] ** self.params["alpha"]) * (
            self.G["heuristic"] ** (-self.params["beta"])
        )

    def build_get(
        self,
        start: int,
        cost_matrix: Optional[np.ndarray] = None,
        proba_matrix: Optional[np.ndarray] = None,
    ):

        if cost_matrix is None:
            cost_matrix = self.G["heuristic"]

        proba_matrix = (
            self.__compute_proba_matrix() if proba_matrix is None else proba_matrix
        )

        cost = 0
        
        available_nodes = np.copy(self.params["mask"][0, :]) if 'mask' in self.params else np.ones(self.G['e'].shape[0],dtype=bool)

        if not available_nodes[start]:
            start = np.random.choice(np.where(available_nodes)[0])
        available_nodes[start] = False

        solution = [start]
        chosen_node = start

        while any(available_nodes):

            chosen_node = self.__choose_next_node(
                available_nodes, chosen_node, proba_matrix
            )
            available_nodes[chosen_node] = False
            cost += cost_matrix[
                chosen_node, solution[-1]
            ]  # Online evaluation of the cost
            solution.append(chosen_node)

        cost += cost_matrix[solution[0], solution[-1]]
        solution.append(solution[0])

        return [solution, cost]
