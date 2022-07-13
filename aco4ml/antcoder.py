import numpy as np
import itertools
from copy import deepcopy


def compute_heuristic(X, y):
    """
    It computes the heuristic for each feature by computing the spread of the conditional means of the
    feature for each class, and normalizing it by the maximum spread

    :param X: the data matrix
    :param y: the class labels
    :return: The heuristic is the ratio of the spread of the feature to the spread of the feature with
    the smallest spread.
    """
    classes = np.unique(y)
    conditionnal_means = np.array(
        [[np.mean(X[y == k, i]) for k in classes] for i in range(X.shape[1])]
    )
    spreads = np.abs(conditionnal_means[:, 0] - conditionnal_means[:, 1])
    return (spreads - np.min(spreads)) / (np.max(spreads) - np.min(spreads))


def Antcoder(model_params: dict, X: np.ndarray, y: np.ndarray):

    params_lens = [len(v) for v in model_params.values()]

    total_len = np.sum(params_lens)
    #%%
    e_phero_matrix = np.ones((X.shape[1] + total_len, X.shape[1] + total_len))
    v_phero_matrix = deepcopy(e_phero_matrix)
    regularization = deepcopy(e_phero_matrix)
    heuristic = deepcopy(e_phero_matrix)
    heuristic[: X.shape[1], : X.shape[1]] = heuristic[
        : X.shape[1], : X.shape[1]
    ] @ np.diag(compute_heuristic(X, y))

    params_lens = [0] + list(np.cumsum(params_lens))
    for n1, n2 in itertools.pairwise(params_lens):
        regularization[
            X.shape[1] + n1 : X.shape[1] + n2, X.shape[1] + n1 : X.shape[1] + n2
        ] = 0

    compt = 0
    id2hp = {}
    for name, params in model_params.items():
        for value_parameter in params:
            id2hp[X.shape[1] + compt] = [name,value_parameter]
            compt += 1

    return {
        "e": e_phero_matrix * 0.5,
        "v": v_phero_matrix * 0.5,
        "heuristic": heuristic,
        "regularization": regularization.astype(bool),
    }, id2hp,params_lens
