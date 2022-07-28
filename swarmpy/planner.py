from typing import Optional, Any
import numpy as np
from .aco_step import ACO_Step

#TODO
# - Use Multi typed ant
# - Make some parameters time Varying

class Planner(ACO_Step):
    def __init__(self, ants_parameters: dict[str, Any]):
        self.ants_parameters = ants_parameters
        if "q" not in self.ants_parameters:
            self.ants_parameters["q"] = 0

        if "type" not in self.ants_parameters:
            self.ants_parameters["type"] = 2

    def run(self, nb_iter: int, G: dict[str, np.ndarray]):
        """
        > The function takes in a graph and returns a dictionary of parameters for the ant colony
        optimization algorithm

        :param nb_iter: number of iterations to run the algorithm
        :type nb_iter: int
        :param G: the graph
        :type G: dict[str, np.ndarray]
        :return: The ants_parameters dictionary is being returned.
        """

        n = G["e"].shape[0]

        if "mask" not in self.ants_parameters:
            self.ants_parameters["mask"] = np.ones((n, n), dtype=bool)

        return {"ants_parameters": [self.ants_parameters]}


class RandomizedPlanner(Planner):
    def __init__(
        self,
        alpha_bounds: list,
        beta_bounds: list,
        gamma_bounds: list,
        ants_parameters: Optional[dict[str, Any]] = None,
    ):
        if ants_parameters is None:
            ants_parameters = {}
        super().__init__(ants_parameters)
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.gamma_bounds = gamma_bounds

    def run(self, nb_iter: int, G: dict[str, np.ndarray]):
        """
        > This function is called once per iteration, and it returns a list of dictionaries, one for
        each ant. Each dictionary contains the parameters for that ant

        :param nb_iter: number of iterations to run the algorithm for
        :type nb_iter: int
        :param G: the graph
        :type G: np.ndarray
        :return: A list of dictionaries, each dictionary containing the parameters for one ant.
        """
        n = G["e"].shape[0]

        if "mask" not in self.ants_parameters:
            self.ants_parameters["mask"] = np.ones((n, n), dtype=bool)

        params = [
            {
                "alpha": np.random.uniform(*self.alpha_bounds),
                "beta": np.random.uniform(*self.beta_bounds),
                "gamma" : np.random.uniform(*self.gamma_bounds),
                **self.ants_parameters,
            }
            
        ] * n
        return {"ants_parameters": params}
