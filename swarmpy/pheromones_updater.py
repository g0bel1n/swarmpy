import warnings
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

from .aco_step import ACO_Step


def get_hyperparameters_from_indexes(
    solutions: List[list], ants_parameters: List[dict], id2hp: dict
) -> List[dict]:
    sklearn_compatible_ants_hyperparameters: List[dict] = []

    # as ant_parameter["n_features_kept"] might not be constant at the colony scale
    for solution, ant_parameter in zip(solutions, ants_parameters):
        hyperparameter = dict(
            [id2hp[_id] for _id in solution[ant_parameter["n_features_kept"] :]]
        )
        sklearn_compatible_ants_hyperparameters.append(hyperparameter)
    return sklearn_compatible_ants_hyperparameters


def filter_uncompatible_hyperparameters(
    index_to_remove: List[int],
    solutions: List[list],
    sklearn_compatible_ants_hyperparameters: List[dict],
    ants_parameters: List[dict],
) -> tuple[list, list, list]:

    solutions = [sol for ant, sol in enumerate(solutions) if ant not in index_to_remove]

    sklearn_compatible_ants_hyperparameters = [
        ant_hyperparameters
        for ant, ant_hyperparameters in enumerate(
            sklearn_compatible_ants_hyperparameters
        )
        if ant not in index_to_remove
    ]
    ants_parameters = [
        ant_parameters
        for ant, ant_parameters in enumerate(ants_parameters)
        if ant not in index_to_remove
    ]

    return solutions, sklearn_compatible_ants_hyperparameters, ants_parameters


# > This class is an abstract base class for updating pheromones
class BasePheromonesUpdater(ABC):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator,
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
        self, solutions: List[list], id2hp: dict, ants_parameters: List[dict]
    ) -> tuple[List[tuple], List[dict]]:

        cv = StratifiedKFold(shuffle=True, n_splits=self.n_splits)
        scores = []

        sklearn_compatible_ants_hyperparameters: List[
            dict
        ] = get_hyperparameters_from_indexes(solutions, ants_parameters, id2hp)

        # Enables not to split n_splits times x len(solutions) the dataset.
        for train_index, test_index in cv.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            score = []
            index_to_remove = []
            for i, (solution, hyperparameter, ant_parameter) in enumerate(
                zip(solutions, sklearn_compatible_ants_hyperparameters, ants_parameters)
            ):
                try:
                    score.append(
                        self.estimator(**hyperparameter)
                        .fit(
                            X_train[:, solution[: ant_parameter["n_features_kept"]]],
                            y_train,
                        )
                        .score(
                            X_test[:, solution[: ant_parameter["n_features_kept"]]],
                            y_test,
                        )
                    )
                except ValueError:
                    # Sometimes ants yield uncompatible hyperparameters combinaison.
                    # e.g. penalty = l1 and solver = liblinear for sklearn.linear_model.LogisticRegression
                    index_to_remove.append(i)

            (
                solutions,
                sklearn_compatible_ants_hyperparameters,
                ants_parameters,
            ) = filter_uncompatible_hyperparameters(
                index_to_remove,
                solutions,
                sklearn_compatible_ants_hyperparameters,
                ants_parameters,
            )

            if not len(scores):
                scores = np.array(score).reshape((len(score), 1))
            else:
                scores += np.array(score).reshape((len(score), 1))

        scores /= float(self.n_splits)  # type: ignore

        evaluated_solutions = list(zip(solutions, scores))
        evaluated_solutions.sort(key=lambda x: x[1], reverse=True)
        return evaluated_solutions, ants_parameters

    @abstractmethod
    def update(
        self, G: dict[str, np.ndarray], evaluated_solutions: List[tuple]
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
        ants_parameters: list,
        id2hp: dict,
    ):
        G = self.evaporate(G=G)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluated_solutions, ants_parameters = self.evaluate(
                solutions, id2hp=id2hp, ants_parameters=ants_parameters
            )
        G = self.update(G=G, evaluated_solutions=evaluated_solutions)

        if self.bounded:
            G["e"][G["e"] > self.bounds[1]] = self.bounds[1]  # type: ignore
            G["e"][G["e"] < self.bounds[0]] = self.bounds[0]  # type: ignore
            G["v"][G["v"] > self.bounds[1]] = self.bounds[1]  # type: ignore
            G["v"][G["v"] < self.bounds[0]] = self.bounds[0]  # type: ignore

        return {
            "G": G,
            "solutions": evaluated_solutions,
            "ants_parameters": ants_parameters,
        }


# It's a pheromones updater that updates pheromones proportionnally to the quality of the solution
class ProportionnalPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def update(self, G: dict[str, np.ndarray], evaluated_solutions: List[tuple]):

        for solution, score in evaluated_solutions:
            for i in solution:
                for j in solution:
                    if i != j:
                        G["e"][i, j] += self.Q * score
                        G["e"][j, i] = G["e"][i, j]
                G["v"][:, i] += self.Q * score

        return G


# "This class is a pheromones updater that updates the pheromones on the k best paths found so far."
class BestSoFarPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator,
        n_splits: int = 5,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[List[float]] = None,
        k: int = 5,
    ) -> None:
        super().__init__(X, y, estimator, n_splits, evaporation_rate, Q, bounds)
        self.bestSoFar = []
        self.k = k

    def update(self, G: dict[str, np.ndarray], evaluated_solutions: List[tuple]):
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
            self.bestSoFar = evaluated_solutions[: self.k]
        else:
            self.bestSoFar = self.bestSoFar + evaluated_solutions[: self.k]
            self.bestSoFar.sort(key=lambda x: x[1], reverse=True)
            self.bestSoFar = self.bestSoFar[: self.k]

        for solution, score in self.bestSoFar:
            for i in solution:
                for j in solution:
                    if i != j:
                        G["e"][i, j] += self.Q * score
                        G["e"][j, i] = G["e"][i, j]
                G["v"][:, i] += self.Q * score

        return G


# "This class is a pheromones updater that updates the pheromones on the k best paths found on the tour."
class BestTourPheromonesUpdater(BasePheromonesUpdater, ACO_Step):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator,
        n_splits: int = 5,
        evaporation_rate: float = 0.6,
        Q: float = 1000,
        bounds: Optional[List[float]] = None,
        k: int = 5,
    ) -> None:
        super().__init__(X, y, estimator, n_splits, evaporation_rate, Q, bounds)
        self.k = k

    def update(self, G: dict[str, np.ndarray], evaluated_solutions: List[tuple]):
        """
        > For each solution in the first k solutions, add Q*score to the edge between each pair of nodes
        in the solution

        :param G: the graph
        :type G: dict[str, np.ndarray]
        :param solutions: List[list]
        :type solutions: List[list]
        :return: The updated graph.
        """

        for solution, score in evaluated_solutions[: self.k]:
            for i in solution:
                for j in solution:
                    if i != j:
                        G["e"][i, j] += self.Q * score
                        G["e"][j, i] = G["e"][i, j]
                G["v"][:, i] += self.Q * score

        return G
