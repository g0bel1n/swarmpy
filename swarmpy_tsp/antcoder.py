import numpy as np
from test_set.read_test_set import read_test_set
import itertools


def compute_distance(positions: np.ndarray) -> np.ndarray:
    """
    It computes the distance between each pair of points in the input array
    
    :param positions: a numpy array of shape (n, 2) where n is the number of points
    :type positions: np.ndarray
    :return: The distance between each pair of points.
    """
    distances = np.eye(positions.shape[0])
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            distances[i, j] = (
                (positions[[i, j], :][0] - positions[[i, j], :][1]) ** 2
            ).sum() ** (0.5)
            distances[j, i] = distances[i, j]
    return distances

def Antcoder(filepath: str =  'test_set/berlin52'):
    """
    > The function `Antcoder` takes a filepath to a test set and returns a dictionary of the graph, and
    the optimal score of the test set
    
    :param filepath: the path to the test set file, defaults to test_set/berlin52
    :type filepath: str (optional)
    :return: The graph G and the optimal score.
    """

    solution, position = read_test_set(filepath)
    distances = compute_distance(position)  
    e_pheromones = np.ones((len(position), len(position)), dtype=float)*0.5

    G = {"e": e_pheromones, "heuristic": distances, 'cost_matrix' : distances}
    opt_score = sum(distances[i-1,j-1] for i, j in itertools.pairwise(solution))
    return G, opt_score

