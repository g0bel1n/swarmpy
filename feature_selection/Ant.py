from array import array
import numpy as np

class Ant :
    def __init__(self, n_features:int) -> None:
        self.k = n_features

    def run(self, proba_matrix, start :int):
        n_tot = proba_matrix.shape[0]
        available_nodes = np.ones(n_tot).astype(bool)
        available_nodes[start] = False

        choosen_node = start
        for _ in range(self.k-1):
            
            probas = proba_matrix[available_nodes,choosen_node]
            probas /= np.sum(probas)
            choosen_node = np.random.choice(np.where(available_nodes)[0], p=probas)
            available_nodes[choosen_node] = False
        
        self.solution = np.where(~available_nodes)[0]

    def get_solution(self, estimator, Xs : list[tuple[array,array]], ys: list[tuple[array,array]]):
        
        scores_list = []
        for (X_train, X_test), (y_train, y_test) in zip(Xs,ys):

            estimator.fit(X_train[self.solution],y_train[self.solution])
            scores_list.append(metric.score(y_test, estimator.predict(X_test[self.solution])))

        return self.solution, np.mean(scores_list)
        
