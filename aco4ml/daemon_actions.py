import numpy as np
import logging
from .aco_step import ACO_Step
from typing import List
logger = logging.getLogger(__name__)
logging.basicConfig(format="[SwarmPy] %(message)s", level=logging.INFO)


def local_search(how: str, solution: list, cost_matrix: np.ndarray):
    """
    > If the 2-opt swap improves the solution, then swap the two cities and update the solution cost

    :param how: the type of local search to use. Currently only 2-opt is implemented
    :type how: str
    :param solution: the solution to be improved
    :type solution: list
    :param cost_matrix: the cost matrix of the problem
    :type cost_matrix: np.ndarray
    :return: The solution is being returned.
    """
    if how == "2-opt":
        for i in range(1, len(solution[0]) - 3):
            improvement = (
                cost_matrix[solution[0][i - 1], solution[0][i]]
                + cost_matrix[solution[0][i + 1], solution[0][i + 2]]
                - (
                    cost_matrix[solution[0][i - 1], solution[0][i + 1]]
                    + cost_matrix[solution[0][i], solution[0][i + 2]]
                )
            )
            if improvement > 0:
                solution[0][i], solution[0][i + 1] = solution[0][i + 1], solution[0][i]

                solution[1] -= improvement

    return solution


# It's a subclass of ACO_Step that defines a set of actions that can be performed by a daemon
class DaemonActions(ACO_Step):
    def __init__(self, how: str = "2-opt", k: int = 10) -> None:

        self.k = k
        self.how = how
        assert how == "2-opt", ValueError("The only daemon action available is 2-opt")
        self.all_unique_solutions = []

    def run(self, G: dict[str, np.ndarray], solutions: List[list]):
        """
        > The function takes a list of solutions and returns it after applying local
        search to each of the k first solution.

        :param G: dict[str, np.ndarray]
        :type G: dict[str, np.ndarray]
        :param solutions: List[list]
        :type solutions: List[list]
        :return: The solutions are being returned.
        """

        
        solutions = [sol for sol in solutions if sol not in self.all_unique_solutions]
        
        self.all_unique_solutions += solutions

        return {"solutions": solutions}
