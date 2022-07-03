import numpy as np
from .aco_step import ACO_Step
from tqdm import tqdm
from typing import Optional


# > This class is an iterator that returns the next step of the ACO algorithm
class ACO_Pipeline(ACO_Step):
    def __init__(
        self,
        steps: list[tuple[str, ACO_Step]],
        verbose=0,
        iter_max = 20,
        as_step=False, 
        last_step = False, 
        metapipeline = False
    ):
        self.steps = steps
        self.verbose = verbose
        self.G: dict[str, np.ndarray]
        self.solutions = []
        self.ant_params: Optional[dict] = None
        self.iter_max: int = iter_max
        self.as_step = False if  last_step else as_step
        self.last_step = last_step
        self.metapipeline = metapipeline

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
        > The function `run` takes as input a graph `G` and a maximum number of iterations `iter_max`
        and returns a list of solutions
        
        :param G: The graph we want to solve
        :param iter_max: the number of iterations to run the algorithm for, defaults to 20 (optional)
        :return: The best solutions found by the algorithm.
        """
        self.G = G
        run_params = {
            "ant_params": self.ant_params,
            "G": self.G,
            "solutions": self.solutions,
            "nb_iter": 0,
        }

        solutions_bank = []
        pbar = tqdm(range(self.iter_max), desc="SwarmPy", ascii="░▒█")
        for i in pbar:
            run_params["nb_iter"] = i
            solutions = self.iter(run_params=run_params)
            solutions_bank.append(solutions[0])
            pbar.set_description(f'SwarmPy |{"Step|" if self.as_step   else "Final Step|" if self.last_step else ""} Score : {solutions[-1][1] if self.metapipeline else solutions[0][1]}')


        if self.as_step : 
            return {"G": self.G}

        elif self.last_step : 
            return {'solutions' : solutions_bank}

        elif self.metapipeline:
            return {'solutions' : run_params['solutions'] }

        return {'solutions' : solutions_bank}
