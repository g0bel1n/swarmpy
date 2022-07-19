# %%
import os

os.chdir('..')

from time import perf_counter

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_selection import RFE
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from swarmpy import *

from hyperparamers_grid import tree_params

#%%

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

cut = int(0.8 * X.shape[0])

X_train, y_train = X[:cut, :], y[:cut, :].reshape(cut)
X_test, y_test = X[cut:, :], y[cut:, :].reshape(X.shape[0]-cut)

n_features_kept = 8

#%%

means, variances, times = [], [], []
for _ in range(10):
    G, id2hp, hp_map = Antcoder(tree_params, X_train, y_train)

    aco_pipeline = ACO_Pipeline(
        [
            (
                "Planner",
                Planner(
                    {
                        "alpha": 1.0,
                        "beta": 2.0,
                        "gamma": 2.0,
                        "n_features_kept": n_features_kept,
                        "n_hp": len(tree_params),
                    }
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
                    evaporation_rate=0.149,
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

    start = perf_counter()
    search_space = {f"estimator__{k}": v for k, v in tree_params.items()}
    pipe = Pipeline(
        [
            (
                "feature_selection",
                RFE(DecisionTreeClassifier(), n_features_to_select=n_features_kept),
            ),
            ("estimator", DecisionTreeClassifier()),
        ]
    )
    clf = GridSearchCV(estimator=pipe, param_grid=search_space)
    clf.fit(X_train, y_train)
    time_classic = perf_counter() - start
    print(f"temps HP : {time_classic}")

    aco_score = cross_val_score(
        lr_aco,
        X_test[:, best_cols],
        y_test,
        cv=10,
        scoring=make_scorer(balanced_accuracy_score),
    )
    not_aco_score = cross_val_score(
        clf.best_estimator_,
        X_test,
        y_test,
        cv=10,
        scoring=make_scorer(balanced_accuracy_score),
    )
    print(f" ACO : {np.mean(aco_score)=} |  {np.std(aco_score)=} ")
    print(f" Not ACO : {np.mean(not_aco_score)=} |  {np.std(not_aco_score)=}")

    means.append([np.mean(aco_score), np.mean(not_aco_score)])
    variances.append([np.std(aco_score), np.std(not_aco_score)])
    times.append([time_aco, time_classic])

df = pd.DataFrame({"means": means, "variances": variances, "times": times})

#%%

df.to_csv("trials.csv")

#%%
np.mean(means, axis=0)

# %%
px.line(means)

#%%
px.line(times)

# %%
rfe.get_feature_names_out()

# %%
best_params

# %%
clf.best_score_

# %%
datapoints = [[x, y] for x, y in enumerate(G["v"][0, : X.shape[1]])]

datapoints.sort(key=lambda x: x[1])

datapoints = np.array(datapoints)
px.bar(
    y=datapoints[:, 1],
    x=datapoints[:, 0].astype(int).astype(str),
    category_orders=datapoints[:, 0].astype(str),
    text_auto=True,
)

# %%
px.imshow(
    (max(rfe.ranking_) - rfe.ranking_.reshape(digits.images[0].shape))
    / (max(rfe.ranking_))
)

# %%
px.imshow(G["v"][0, : X.shape[1]].reshape(digits.images[0].shape))

# %%


X.shape[1]
# %%
