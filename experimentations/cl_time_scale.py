from time import perf_counter

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_selection import RFE
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from swarmpy import *
from hyperparamers_grid import tree_params

X = (
    pd.read_csv("data/sylvine/sylvine_train.data", header=None, sep=" ")
    .dropna(axis=1)
    .to_numpy()
)

y = (
    pd.read_csv("data/sylvine/sylvine_train.solution", header=None, sep=" ")
    .dropna(axis=1)
    .to_numpy()
).reshape(len(X))
n_features_kept =10
times = []
len_params = []
aco_score = []
differently_sized_params = [{k:tree_params[k] for k in list(tree_params.keys())[:i]} for i in range(1,len(tree_params))]
for param_grid in differently_sized_params : 
    print(param_grid)
    G, id2hp, hp_map = Antcoder(param_grid, X, y)

    aco_pipeline = ACO_Pipeline(
        [
            (
                "Planner",
                RandomizedPlanner(
                    alpha_bounds=[1.0, 2.0],
                    beta_bounds=[1.0, 1.0],
                    gamma_bounds=[1.0, 3.0],
                    ants_parameters={
                        "n_features_kept": n_features_kept,
                        "n_hp": len(param_grid),
                    },
                ),
            ),
            ("Sol", SolutionConstructor(hp_map=hp_map)),
            ("DA", DaemonActions()),
            (
                "Updater",
                BestSoFarPheromonesUpdater(
                    X=X,
                    y=y,
                    estimator=DecisionTreeClassifier,
                    bounds=[0.1, 5],
                    Q=0.05,
                    evaporation_rate=0.1,
                ),
            ),
        ],
        n_iter=20,
        id2hp=id2hp,
    )

    start = perf_counter()
    solutions = aco_pipeline.run(G)["solutions"]
    best_sol = max(solutions, key=lambda x: x[1])[0]
    best_params = dict([id2hp[_id] for _id in best_sol[n_features_kept:]])
    best_cols = best_sol[:n_features_kept]
    time_aco = perf_counter() - start
    lr_aco = DecisionTreeClassifier(**best_params).fit(X[:, best_cols], y)
    print(time_aco)
    times.append(time_aco)
    len_params.append(np.prod([len(param) for param in param_grid.values()]))
    aco_score += [balanced_accuracy_score(y, lr_aco.predict(X[:, best_cols]))]
print(aco_score)
print(f" ACO : {np.mean(aco_score)=} |  {np.std(aco_score)=} ")

fig = px.line(x=len_params, y=times)
fig.update_layout(xaxis_title ="Nombre de combinaisons possibles", yaxis_title = "Temps d'execution")

fig.show()
