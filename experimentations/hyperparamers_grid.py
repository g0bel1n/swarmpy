rfc  = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

log_reg_params  = {'penalty' : ['l1', 'none','l2', 'elasticnet'], 'C' : [i/10 for i in range(1,10)], 'max_iter' : [100, 200, 300]}

tree_params = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "min_samples_split": list(range(4, 10)),
    "min_samples_leaf": list(range(1, 5)),
    "max_features": ["auto", "log2", "sqrt"],
}