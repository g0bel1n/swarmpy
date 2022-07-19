import numpy as np
import logging
from .aco_step import ACO_Step
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(format="[SwarmPy] %(message)s", level=logging.INFO)


# It's a subclass of ACO_Step that defines a set of actions that can be performed by a daemon
class DaemonActions(ACO_Step):
    def __init__(self) -> None:
        self.all_unique_solutions = []

    def run(

        self, G: dict[str, np.ndarray], solutions: List[list], ants_parameters: list
    ):
        """
        > This function takes in a list of solutions and a list of ants parameters, and returns a list
        of solutions and a list of ants parameters that have never been evaluated so far. It avoids training several times for the same solution.
        
        :param G: dict[str, np.ndarray]
        :type G: dict[str, np.ndarray]
        :param solutions: list of solutions
        :type solutions: List[list]
        :param ants_parameters: list of lists, each list contains the parameters of an ant
        :type ants_parameters: list
        """

        clean_sol, clean_ants_params = [], []
        for i in range(len(solutions)):
            if solutions[i] not in self.all_unique_solutions:
                clean_ants_params.append(ants_parameters[i])
                clean_sol.append(solutions[i])

        self.all_unique_solutions += clean_sol

        return {"solutions": clean_sol, "ants_parameters": clean_ants_params}
