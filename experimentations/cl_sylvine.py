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
)

n_features_kept = 10
skf = StratifiedKFold(shuffle=True)
aco_score = []
not_aco_score = []


for train_index, test_index in skf.split(X, y):
    X_train, y_train = X[train_index], y[train_index].reshape(len(train_index))
    X_test, y_test = X[test_index], y[test_index].reshape(len(test_index))

    rfe = RFE(estimator = DecisionTreeClassifier(), n_features_to_select=n_features_kept).fit(X_train,y_train)

    clf = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_params, refit=True).fit(
            X_train[:,support:=rfe.get_support()], y_train
    )

    G, id2hp, hp_map = Antcoder(tree_params, X_train, y_train)

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
                        "n_hp": len(tree_params),
                    },
                ),
            ),
            ("Sol", SolutionConstructor(hp_map=hp_map)),
            ("DA", DaemonActions()),
            (
                "Updater",
                BestSoFarPheromonesUpdater(
                    X=X_train,
                    y=y_train,
                    estimator=DecisionTreeClassifier,
                    bounds=[0.1, 5],
                    Q=0.05,
                    evaporation_rate=0.3,
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
    lr_aco = DecisionTreeClassifier(**best_params).fit(X_train[:, best_cols], y_train)
    print(time_aco)

    aco_score += [balanced_accuracy_score(y_test, lr_aco.predict(X_test[:, best_cols]))]
    not_aco_score += [
        balanced_accuracy_score(y_test, clf.best_estimator_.predict(X_test[:, support]))
]

print(aco_score)
print(not_aco_score)
print(f" ACO : {np.mean(aco_score)=} |  {np.std(aco_score)=} ")
print(f" Not ACO : {np.mean(not_aco_score)=} |  {np.std(not_aco_score)=}")

fig = px.line(y=[aco_score, not_aco_score], x=list(range(5)), labels=["ACO", "Not ACO"])
fig.show()
