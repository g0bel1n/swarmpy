import numpy as np
import itertools


def normalized_interclass_absolute_spread(X, y):
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


def Antcoder(hyperparameters_grid: dict, X: np.ndarray, y: np.ndarray):

    hyperparameters_grid_sizes = [len(v) for v in hyperparameters_grid.values()]

    total_len = np.sum(hyperparameters_grid_sizes)

    e_phero_matrix = np.ones((X.shape[1] + total_len, X.shape[1] + total_len))
    v_phero_matrix = np.ones_like(e_phero_matrix)
    regularization = np.ones_like(e_phero_matrix)
    heuristic = np.ones_like(e_phero_matrix)

    heuristic[: X.shape[1], : X.shape[1]] = heuristic[
        : X.shape[1], : X.shape[1]
    ] @ np.diag(normalized_interclass_absolute_spread(X, y))

    hyperparameters_index_bounds = [0] + list(
        np.cumsum(hyperparameters_grid_sizes)
    )  # if there is k parameter 1 and n parameter 2, then hyperparameters_index_bounds = [0, k, n+k]

    for n1, n2 in itertools.pairwise(hyperparameters_index_bounds):
        regularization[
            X.shape[1] + n1 : X.shape[1] + n2, X.shape[1] + n1 : X.shape[1] + n2
        ] = 0

    compt = 0

    # Converts the index of an hyperparameter value to {'parameter_1' : parameter_value}
    # It is for compatibility with scikit-learn estimators
    id2hp = {}
    for hyperparameter_name, hyperparameter_values in hyperparameters_grid.items():
        for hyperparameter_value in hyperparameter_values:
            id2hp[X.shape[1] + compt] = [hyperparameter_name, hyperparameter_value]
            compt += 1

    return (
        {
            "e": e_phero_matrix * 0.5,
            "v": v_phero_matrix * 0.5,
            "heuristic": heuristic,
            "regularization": regularization.astype(bool),
        },
        id2hp,
        hyperparameters_index_bounds,
    )
