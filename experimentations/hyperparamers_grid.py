rfc = {
    "n_estimators": [200, 500],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [4, 5, 6, 7, 8],
    "criterion": ["gini", "entropy"],
}

log_reg_params = {
    "penalty": ["l1", "none", "l2", "elasticnet"],
    "C": [i / 10 for i in range(1, 10)],
    "max_iter": [100, 200, 300],
}

tree_params = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "min_samples_split": list(range(4, 10)),
    "min_samples_leaf": list(range(1, 5)),
    "max_features": ["auto", "log2", "sqrt"],
    "max_depth": list(range(3, 10)),
    "max_leaf_nodes": list(range(2, 7))+[None],
}

svm_params = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["linear"],
}
