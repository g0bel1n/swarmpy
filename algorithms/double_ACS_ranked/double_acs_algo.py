import itertools
from threading import Thread

import numpy as np
from tqdm import tqdm

from algorithms.ACS_ranked.acs_ant import ACS_Ant

from algorithms import ACS_Algo

from algorithms.BaseACO import compute_distance


class double_ACS_ranked_algo(ACS_Algo):
    def __init__(
        self,
        positions: np.ndarray,
        q: float,
        iter_max: int = 100,
        rho: float = 0.6,
        Q: float = 100,
    ):
        super().__init__(positions, q, iter_max, rho, Q)

    def fit(self):
        e_pheromones = np.ones((self.batchSize, self.batchSize)) * self.tau_e_0
        heuristics = compute_distance(self.positions)

        cut = int(self.batchSize / 2)
        # self.G_top = {"e": e_pheromones[:cut,:cut], "heuristic": heuristics[:cut,:cut]}
        # self.bottom = {"e": e_pheromones[cut:,cut:], "heuristic": heuristics[cut:,cut:]}
        demi_iter = int(self.iter_max / 2)

        col1 = ACS_Algo(positions=self.positions[:cut], q=0.5, iter_max=demi_iter).fit()
        col2 = ACS_Algo(positions=self.positions[cut:], q=0.5, iter_max=demi_iter).fit()

        e_pheromones[:cut, :cut] = col1.G["e"]
        e_pheromones[cut:, cut:] = col2.G["e"]
        self.G = {"e": e_pheromones, "heuristic": heuristics}
        nb_iter = demi_iter
        self.costs = [a+b for a,b in zip(col1.costs, col2.costs)]
        with tqdm(
            total=demi_iter,
            desc="final run",
            ascii="░▒█",
        ) as pbar:
            while nb_iter < self.iter_max:
                self._BaseACO__construct_solutions()
                self.costs.append(self.solutions[0][1])
                self._BaseACO__update_pheromones()
                # self.__daemon_actions()
                nb_iter += 1
                pbar.update(1)
