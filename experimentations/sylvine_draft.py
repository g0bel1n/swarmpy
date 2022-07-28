# %%
import os

os.chdir("..")

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

# X = np.random.uniform(size=50000).reshape((1000, 50))
# y = (
#     (np.sum([X[:, i] ** (i + 1) for i in range(1, 30) if i % 2 == 0], axis=0) > 1.0)
#     .astype(int)
#     .reshape(1000, 1)
# )


n_features_kept = 10

#%%

    # search_space = {f"estimator__{k}": v for k, v in tree_params.items()}
    # pipe = Pipeline(
    #     [
    #         (
    #             "feature_selection",
    #             RFE(DecisionTreeClassifier(), n_features_to_select=n_features_kept),
    #         ),
    #         ("estimator", DecisionTreeClassifier()),
    #     ]
    # )
    # clf = GridSearchCV(estimator=pipe, param_grid=search_space)
    # clf.fit(X_train, y_train)

#%%
    
skf = StratifiedKFold(shuffle=True)
aco_score = []
not_aco_score = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index].reshape(len(train_index)), y[test_index].reshape(len(test_index))
    start = perf_counter()
    rfe = RFE(
        DecisionTreeClassifier(), n_features_to_select=n_features_kept, step=1
    ).fit(X_train, y_train)
    support = rfe.get_support()
    clf = GridSearchCV(
        estimator=DecisionTreeClassifier(), param_grid=tree_params, refit=True
    )
    clf.fit(X_train[:, support], y_train)
    time_classic = perf_counter() - start
    print(f"temps HP : {time_classic}")

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

    
    aco_score += [balanced_accuracy_score(y, lr_aco.predict(X[:, best_cols]))]
    not_aco_score += [balanced_accuracy_score(y, clf.best_estimator_.predict(X[:, support]))]


print(f" ACO : {np.mean(aco_score)=} |  {np.std(aco_score)=} ")
print(f" Not ACO : {np.mean(not_aco_score)=} |  {np.std(not_aco_score)=}")

px.line(y=[aco_score, not_aco_score], x=list(range(5)),labels = ['ACO','Not ACO'])

#%%
# %%

#%%
px.line(times)

# %%
rfe.get_feature_names_out()

# %%
best_params

# %%
pipe.fit(X_train, y_train)
np.mean(cross_val_score(pipe, X, y, scoring=make_scorer(balanced_accuracy_score)))
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
