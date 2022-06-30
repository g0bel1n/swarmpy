import numpy as np
from threading import Thread
from .Ants import AS_Ant, ACS_Ant
from .Ants.base_ant import BaseAnt

from .aco_step import ACO_Step


def add_solution_to_list(solutions: list, start: int, Ant: BaseAnt, **kwargs):
    ant = Ant(**kwargs)  # type: ignore
    sol = list(ant.build_get(start))
    solutions.append(sol)


class SolutionConstructor(ACO_Step):
    def __init__(self, how: str = "AS") -> None:
        self.Ant = ACS_Ant if how == "ACS" else AS_Ant

    def __repr__(self) -> str:
        return "SolutionConstructor"

    def run(
        self,
        G: dict[str, np.ndarray],
        ant_params: dict[str, np.ndarray],
        solutions: list,
    ):

        threads = []
        for i in range(G["e"].shape[0]):
            threads.append(
                Thread(
                    target=add_solution_to_list,
                    args=(solutions, i, self.Ant),
                    kwargs={"G": G, "ant_params": ant_params},
                )
            )
            threads[-1].start()

        for thread in threads:
            thread.join()

        solutions.sort(key=lambda x: x[1])

        return {"solutions": solutions}
