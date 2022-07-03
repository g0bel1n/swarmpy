from .base_ant import BaseAnt
import numpy as np


class AS_Ant(BaseAnt):
    def _BaseAnt__choose_next_node(
        self, available_nodes: np.ndarray, chosen_node: int, proba_matrix: np.ndarray
    ) -> int:
        probas = proba_matrix[available_nodes, chosen_node]
        probas /= np.sum(probas)
        return np.random.choice(np.where(available_nodes)[0], p=probas)
