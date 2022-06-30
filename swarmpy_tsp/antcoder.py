import numpy as np
from test_set.read_test_set import read_test_set
import itertools


def compute_distance(positions: np.ndarray) -> np.ndarray:
    distances = np.eye(positions.shape[0])
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            distances[i, j] = (
                (positions[[i, j], :][0] - positions[[i, j], :][1]) ** 2
            ).sum() ** (0.5)
            distances[j, i] = distances[i, j]
    return distances

def Antcoder(filepath: str =  'test_set/berlin52'):

    solution, position = read_test_set(filepath)
    distances = compute_distance(position)  
    e_pheromones = np.ones((len(position), len(position)), dtype=float)

    G = {"e": e_pheromones, "heuristic": distances, 'cost_matrix' : distances}
    opt_score = sum(distances[i-1,j-1] for i, j in itertools.pairwise(solution))
    return G, opt_score

