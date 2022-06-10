#%%
import itertools
from threading import Thread
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from Ant import Ant
data = load_iris()

# %%
def compute_heuristic(X,y):
    classes = np.unique(y)
    conditionnal_means = [[np.mean(X[y==k,i]) for k in classes] for i in range(X.shape[1])]
    spreads = np.max(conditionnal_means, axis=0)-np.min(conditionnal_means, axis=0)
    return (spreads - np.min(spreads))/(np.max(spreads)- np.mean(spreads))

def single_ant_solution_construction(start, proba_matrix, estimator, metric, solutions, n_features, Xs, ys):
    ant = Ant(start, n_features)
    ant.run(proba_matrix)
    solutions.append(ant.get_solution(estimator, metric, Xs, ys))


class ACO_Selector(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    
    def __init__(self, estimator, n_features, iter_max =50, batchSize = 20) -> None:
        super().__init__()
        self.estimator = estimator
        self.n_features = n_features
        self.iter_max = iter_max
        self.best_solution = []
        self.batchSize : int

    def fit(self, X, y):
        self.batchSize = X.shape[1]
        e_pheromones = np.ones((self.batchSize, self.batchSize))*self.tau_e_0
        v_pheromones = np.ones(self.batchSize)*self.tau_e_0
        heuristic = compute_heuristic(X,y)
        self.G = {'e':e_pheromones, 'v': v_pheromones, 'h': heuristic }
        self.solutions = []
        nb_iter = 0
        while nb_iter < self.iter_max:
            self._construct_solutions()
            self._update_pheromones()
            if self.solutions[1]> self.best_solution[1]: self.best_solution = self.solutions
            nb_iter+=1 

    def _construct_solutions(self):
        solutions = []
        proba_matrix = (self.G['e']**self.alpha) @ self.diag((self.G['v']**self.gamma)*(self.G['h'])**self.beta)
        threads = [Thread(target=single_ant_solution_construction, args=(start = i, proba_matrix=proba_matrix, estimator = estimator, metric = metric, solutions = solutions, n_features = self.n_features)).start() for i in range(self.batchSize)]
        for thread in threads : thread.stop()
        self.solutions = solutions.sort(key = lambda x: x[1],reverse=True)

    def _update_pheromones(self):
        self.G['e'] = max((1-self.rho_e)*self.G['e'], self.e_pheromone_min)
        self.G['v'] = max((1-self.rho_v)*self.G['v'], self.v_pheromone_min)

        for (i, (solution, cost)), l in itertools.product(enumerate(self.solutions), range(self.n_features)):
            if l+1 < self.n_features : 
                self.G['e'][solution[l],solution[l+1]] = min(self.Q_e*(self.n_features-i)/(cost*self.n_features) + self.G['e'][solution[l],solution[l+1]], self.e_pheromone_max)
            self.G['v'][solution[l]] = min(self.Q_v*(self.n_features-i)/(cost*self.n_features)+self.G['v'][solution[l]], self.v_pheromone_max)