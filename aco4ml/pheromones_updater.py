import itertools
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

from .aco_step import ACO_Step

import warnings 


# > This class is an abstract base class for updating pheromones
class BasePheromonesUpdater(ABC):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator: BaseEstimator,
        n_splits: int = 5,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[List[float]] = None,
    ) -> None:
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.bounds = bounds
        self.n_splits = n_splits
        self.X = X
        self.y = y
        self.estimator = estimator

        if bounds is not None:
            self.bounds.sort()  # type: ignore
            self.bounded = True
        else:
            self.bounded = False

    def evaluate(
        self, solutions: List[list], id2hp: dict, ant_params: list
    ) -> List[list]:
        cv = StratifiedKFold(shuffle=True, n_splits=self.n_splits)
        scores = None

        params: List[dict] = []
        for solution, ant_param in zip(solutions, ant_params):
            intermediate_list = dict(
                [id2hp[_id] for _id in solution[ant_param["n_feature_obj"] :]]
            )
            params.append(intermediate_list)

        for train_index, test_index in cv.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            score = []
            index_to_remove = []
            for i, (solution, param, ant_param) in enumerate(zip(solutions, params, ant_params)):
                try:
                    score.append(
                        self.estimator(**param)
                        .fit(
                            X_train[:, solution[: ant_param["n_feature_obj"]]], y_train
                        )
                        .score(
                            X_test[:, solution[: ant_param["n_feature_obj"]]], y_test
                        )
                    )
                except ValueError:
                    index_to_remove.append(i)

            assert len(index_to_remove)+len(score) == len(solutions), 'eyoooo'
    
            solutions = [sol for i,sol in enumerate(solutions) if i not in index_to_remove]
            params = [sol for i,sol in enumerate(params) if i not in index_to_remove]
            ant_params = [sol for i,sol in enumerate(ant_params) if i not in index_to_remove]

            if scores is None:
                scores = np.array(score).reshape((len(score), 1))
            else:
                scores += np.array(score).reshape((len(score), 1))

        assert len(scores) == len(solutions), "error here buddy"
        scores /= self.n_splits

        solutions = list(zip(solutions, scores))
        solutions.sort(key=lambda x: x[1], reverse=True)
        return solutions, ant_params

    @abstractmethod
    def update(
        self, G: dict[str, np.ndarray], solutions: List[list]
    ) -> dict[str, np.ndarray]:
        pass

    def evaporate(self, G: dict[str, np.ndarray]):

        G["e"] *= 1 - self.evaporation_rate
        G["v"] *= 1 - self.evaporation_rate
        return G

    def run(
        self,
        G: dict[str, np.ndarray],
        solutions: List[list],
        ant_params: list,
        id2hp: dict,
        n_features: int,
    ):
        G = self.evaporate(G=G)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solutions, ant_params = self.evaluate(
                solutions, id2hp=id2hp, ant_params=ant_params
            )
        G = self.update(G=G, solutions=solutions)

        if self.bounded:
            G["e"][G["e"] > self.bounds[1]] = self.bounds[1]  # type: ignore
            G["e"][G["e"] < self.bounds[0]] = self.bounds[0]  # type: ignore
            G["v"][G["v"] > self.bounds[1]] = self.bounds[1]  # type: ignore
            G["v"][G["v"] < self.bounds[0]] = self.bounds[0]  # type: ignore

        return {"G": G, "solutions": solutions, "ant_params": ant_params}


# It's a pheromones updater that updates pheromones proportionnally to the quality of the solution
class ProportionnalPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def update(self, G: dict[str, np.ndarray], solutions: List[list]):

        for solution, score in solutions:
            for i, j in itertools.pairwise(solution):
                G["e"][i, j] += self.Q * score
                G["e"][j, i] = G["e"][i, j]
                G["v"][:, i] += self.Q * score
            G["v"][:, j] += self.Q * score

        return G


# "This class is a pheromones updater that updates the pheromones on the k best paths found so far."
class BestSoFarPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator: BaseEstimator,
        n_splits: int = 5,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[List[float]] = None,
        k: int = 5,
    ) -> None:
        super().__init__(X, y, estimator, n_splits, evaporation_rate, Q, bounds)
        self.bestSoFar = []
        self.k = k

    def update(self, G: dict[str, np.ndarray], solutions: List[list]):
        """
        > For each solution in the list of solutions, add the value of Q multiplied by the score of the
        solution to the edge weights of the graph

        :param G: the graph
        :type G: dict[str, np.ndarray]
        :param solutions: List[list]
        :type solutions: List[list]
        :return: The updated graph.
        """
        if len(self.bestSoFar) == 0:
            self.bestSoFar = solutions[: self.k]
        else:
            self.bestSoFar = self.bestSoFar + solutions[: self.k]
            self.bestSoFar.sort(key=lambda x: x[1], reverse=True)
            self.bestSoFar = self.bestSoFar[: self.k]

        for solution, score in self.bestSoFar:
            for i, j in itertools.pairwise(solution):
                G["e"][i, j] += self.Q * score
                G["e"][j, i] = G["e"][i, j]
                G["v"][:, i] += self.Q * score
            G["v"][:, j] += self.Q * score
        return G


# "This class is a pheromones updater that updates the pheromones on the k best paths found on the tour."
class BestTourPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator: BaseEstimator,
        n_splits: int = 5,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[List[float]] = None,
        k: int = 5,
    ) -> None:
        super().__init__(X, y, estimator, n_splits, evaporation_rate, Q, bounds)
        self.k = k

    def update(self, G: dict[str, np.ndarray], solutions: List[list]):
        """
        > For each solution in the first k solutions, add Q*score to the edge between each pair of nodes
        in the solution

        :param G: the graph
        :type G: dict[str, np.ndarray]
        :param solutions: List[list]
        :type solutions: List[list]
        :return: The updated graph.
        """

        for solution, score in solutions[: self.k]:
            for i, j in itertools.pairwise(solution):
                G["e"][i, j] += self.Q * score
                G["e"][j, i] = G["e"][i, j]
                G["v"][:, i] += self.Q * score
            G["v"][:, j] += self.Q * score

        return G
