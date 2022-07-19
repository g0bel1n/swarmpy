import numpy as np
from .aco_step import ACO_Step
from tqdm import tqdm
from typing import Optional, List
import logging


# > This class is an iterator that returns the next step of the ACO algorithm
class ACO_Pipeline(ACO_Step):
    def __init__(
        self,
        steps: List[tuple[str, ACO_Step]],
        id2hp: dict,
        verbose=0,
        n_iter=20,
    ):
        self.steps = steps
        logging.basicConfig(level=verbose)
        self.G: dict[str, np.ndarray]
        self.solutions = []
        self.ants_parameters: Optional[dict] = None
        self.n_iter: int = n_iter
        self.id2hp = id2hp

        self.n_features: int

    def iter(self, run_params: dict[str, np.ndarray]):
        """
        > For each step in the pipeline, run the step with the arguments that it needs, and then update
        the run_params dictionary with the output of the step

        :param run_params: dict[str, np.ndarray]
        :type run_params: dict[str, np.ndarray]
        :return: The solutions to the problem.
        """

        for _, step in self.steps:
            step_args = {
                el: run_params[el] for el in step.get_run_args() if el != "self"
            }
            _out = step.run(**step_args)

            for k, v in _out.items():
                run_params[k] = v
        return run_params["solutions"]

    def run(self, G):
        """
        > The function `run` takes as input a graph `G` and a maximum number of iterations `n_iter`
        and returns a list of solutions

        :param G: The graph we want to solve
        :param n_iter: the number of iterations to run the algorithm for, defaults to 20 (optional)
        :return: The best solutions found by the algorithm.
        """
        self.G = G
        self.n_features = G["e"].shape[0] - len(self.id2hp)
        run_params = {
            "ants_parameters": self.ants_parameters,
            "G": self.G,
            "solutions": self.solutions,
            "nb_iter": 0,
            "id2hp": self.id2hp,
            "n_features": self.n_features,
        }

        solutions_bank = []
        pbar = tqdm(range(self.n_iter), desc="SwarmPy", ascii="░▒█")
        for i in pbar:
            run_params["nb_iter"] = i
            solutions = self.iter(run_params=run_params)
            solutions_bank.append(solutions[0])
            pbar.set_description(
                f"SwarmPy | Score : {max(solutions_bank, key=lambda x: x[1])[1]}"
            )

        return {"solutions": solutions_bank}
