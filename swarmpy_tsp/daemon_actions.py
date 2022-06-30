import numpy as np
import logging
from .aco_step import ACO_Step
import itertools

logger = logging.getLogger(__name__)
logging.basicConfig(format="[SwarmPy] %(message)s", level=logging.INFO)


def local_search(how: str, solution: list, cost_matrix: np.ndarray):
    if how == "2-opt":
        for i in range(1, len(solution[0]) - 3):
            improvement = (
                cost_matrix[solution[0][i - 1], solution[0][i]]
                + cost_matrix[solution[0][i + 1], solution[0][i + 2]]
                - (cost_matrix[solution[0][i - 1], solution[0][i + 1]] + cost_matrix[solution[0][i], solution[0][i + 2]])
            )
            if improvement > 0:
                solution[0][i], solution[0][i + 1] = solution[0][i + 1], solution[0][i]
                
                solution[1] -= improvement

    return solution


class DaemonActions(ACO_Step):
    def __init__(self, how: str = "2-opt", k: int = 10) -> None:

        self.k = k
        self.how = how
        assert how == "2-opt", ValueError("The only daemon action available is 2-opt")

    def run(self, G: dict[str, np.ndarray], solutions: list[list]):
        if self.k > len(solutions):
            self.k = len(solutions)
            logging.info("k > len(solutions), therefore k is set to len(solutions)")

        for j in range(self.k):
            solutions[j] = local_search(
                how=self.how, solution=solutions[j], cost_matrix=G["cost_matrix"]
            )

        solutions.sort(key=lambda x: x[1])

        return {"solutions": solutions}
