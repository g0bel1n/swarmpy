from typing import Optional, Any
import numpy as np
from .aco_step import ACO_Step


class Planner(ACO_Step):
    def __init__(self, ant_params: dict[str, Any]):
        self.ant_params = ant_params
        if 'q' not in self.ant_params:
            self.ant_params['q'] = 0
        


    def run(self, nb_iter: int, G: dict[str, np.ndarray]):
        """
        > The function takes in a graph and returns a dictionary of parameters for the ant colony
        optimization algorithm
        
        :param nb_iter: number of iterations to run the algorithm
        :type nb_iter: int
        :param G: the graph
        :type G: dict[str, np.ndarray]
        :return: The ant_params dictionary is being returned.
        """

        n = G["e"].shape[0]

        if 'mask' not in self.ant_params : 
            self.ant_params['mask'] = np.ones((n,n), dtype=bool)

        return {"ant_params": [self.ant_params]}



class RandomizedPlanner(Planner):
    def __init__(self, alpha_bounds: list, beta_bounds: list, ant_params: Optional[dict[str, Any]] = None):
        if ant_params is None:
            ant_params = {}
        super().__init__(ant_params)
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds

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

        if 'mask' not in self.ant_params : 
            self.ant_params['mask'] = np.ones((n,n), dtype=bool)


        params = [
            {
                "alpha": np.random.uniform(*self.alpha_bounds),
                "beta": np.random.uniform(*self.beta_bounds),
                'q' : self.ant_params['q'],
                'mask' : self.ant_params['mask']
            }
            for _ in range(n)
        ]
        return {"ant_params": params}
