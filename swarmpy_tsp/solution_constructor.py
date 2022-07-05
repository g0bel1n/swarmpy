from threading import Thread
from typing import List

import numpy as np

from .aco_step import ACO_Step
from .Ants import ACS_Ant
from .Ants.base_ant import BaseAnt


def add_solution_to_list(solutions: list, start: int, Ant: BaseAnt, **kwargs):
    """
    It creates an ant of the given type, and then adds the solution it finds to the list of solutions

    :param solutions: list
    :type solutions: list
    :param start: the starting node
    :type start: int
    :param Ant: The class of ant to use
    :type Ant: BaseAnt
    """

    ant = Ant(**kwargs)  # type: ignore
    sol = list(ant.build_get(start))
    solutions.append(sol)


# This class is a subclass of the ACO_Step class, and it is used to construct the solution.
class SolutionConstructor(ACO_Step):
    def __init__(self) -> None:
        self.Ant = ACS_Ant

    def __repr__(self) -> str:
        return "SolutionConstructor"

    def run(
        self,
        G: dict[str, np.ndarray],
        ant_params: List[dict[str, np.ndarray]],
        solutions: list,
    ):
        """
        > The function takes in a graph, a list of parameters for each ant, and a list of solutions. It
        then creates a thread for each ant, and runs the ant algorithm on each thread. The solutions are
        then sorted and returned

        :param G: dict[str, np.ndarray]
        :type G: dict[str, np.ndarray]
        :param ant_params: List[dict[str, np.ndarray]]
        :type ant_params: List[dict[str, np.ndarray]]
        :param solutions: list
        :type solutions: list
        :return: A dictionary with a key of "solutions" and a value of the list of solutions.
        """
        if len(ant_params) == 1:
            ant_params *= np.sum(ant_params[0]["mask"][0, :].astype(int))

        threads = []
        for i, params in enumerate(ant_params):
            threads.append(
                Thread(
                    target=add_solution_to_list,
                    args=(solutions, i, self.Ant),
                    kwargs={"G": G, "ant_params": params},
                )
            )
            threads[-1].start()

        for thread in threads:
            thread.join()

        solutions.sort(key=lambda x: x[1])

        return {"solutions": solutions}
