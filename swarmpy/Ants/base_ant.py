import itertools
from abc import ABC, abstractmethod

import numpy as np


class BaseAnt(ABC):
    def __init__(
        self,
        ant_parameters: dict,
        G: dict[str, np.ndarray],
        n_features: int,
        hp_map: list,
    ) -> None:
        self.ant_parameters = ant_parameters
        self.G = G
        self.n_features = n_features
        self.hp_map = hp_map

    @abstractmethod
    def __choose_next_node(
        self, available_nodes: np.ndarray, chosen_node: int, proba_matrix: np.ndarray
    ) -> int:
        pass

    def __compute_proba_matrix(self) -> np.ndarray:

        return (
            (self.G["e"] ** self.ant_parameters["alpha"])
            * (self.G["heuristic"] ** (self.ant_parameters["beta"]))
            * (self.G["v"] ** self.ant_parameters["gamma"])
        )

    def build_get(
        self,
        start: int,
        best_sol: list,
    ):

        proba_matrix = self.__compute_proba_matrix()

        available_nodes = np.ones(self.G["e"].shape[0], dtype=bool)
        if self.ant_parameters["type"] == 0:
            available_nodes[best_sol[self.n_features :]] = False
        elif self.ant_parameters["type"] == 1:
            available_nodes[best_sol[: self.n_features]] = False

        if not available_nodes[start]:
            start = np.random.choice(np.where(available_nodes)[0])
        available_nodes[start] = False

        solution = [start]
        chosen_node = start
        if chosen_node >= self.n_features:
            for a, b in itertools.pairwise(self.hp_map):
                if (
                    chosen_node - self.n_features >= a
                    and chosen_node - self.n_features < b
                ):
                    available_nodes[a + self.n_features : b + self.n_features] = False
                    break

        while any(available_nodes):

            chosen_node = self.__choose_next_node(
                available_nodes, chosen_node, proba_matrix
            )
            if chosen_node >= self.n_features:
                for a, b in itertools.pairwise(self.hp_map):
                    if (
                        chosen_node - self.n_features >= a
                        and chosen_node - self.n_features < b
                    ):
                        available_nodes[
                            a + self.n_features : b + self.n_features
                        ] = False
                        break

            available_nodes[chosen_node] = False

            if (
                np.sum((~available_nodes[: self.n_features]).astype(int))
                == self.ant_parameters["n_features_kept"]
            ):
                available_nodes[: self.n_features] = False

            solution.append(chosen_node)

        assert (
            len(solution)
            == self.ant_parameters["n_hp"] + self.ant_parameters["n_features_kept"]
        )
        solution.sort()  # Archi important pour comparer les solutions
        return solution
