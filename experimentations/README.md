# %%
import os
from copy import deepcopy
from tkinter import Y

import matplotlib.pyplot as plt
import numpy as np
import warnings
from time import perf_counter

import pandas as pd
import plotly.express as px
from aco4ml import *
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer


# %%
digits = load_digits()
#%%

X = pd.read_csv('sylvine/sylvine_train.data', header=None, sep=' ').dropna(axis=1).to_numpy()
y= pd.read_csv('sylvine/sylvine_train.solution', header=None, sep=' ').dropna(axis=1).to_numpy()

cut = int(.8 * X.shape[0])

X_train, y_train = X[:cut,:],y[:cut,:].reshape(cut)
X_test, y_test = X[cut:,:],y[cut:,:].reshape(cut)
# %%

# params  = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }

#params  = {'penalty' : ['l1', 'none','l2', 'elasticnet'], 'C' : [i/10 for i in range(1,10)], 'max_iter' : [100, 200, 300]}
params = {'criterion': ["gini", "entropy"] , 'splitter' : ['best', 'random'], 'min_samples_split': list(range(4,10)), 'min_samples_leaf' : list(range(1,5)), 'max_features' : ['auto', 'log2', 'sqrt']}



n_feature_obj = 8

# %%
y_train
#%%

means, variances, times = [], [], []
for _ in range(10) : 
    G, id2hp, hp_map = Antcoder(params,X_train, y_train)

    aco_pipeline = ACO_Pipeline(
        [
            ("Planner", Planner({"alpha": 1.0, "beta": 2.0, 'gamma' : 2.0, 'n_feature_obj' : n_feature_obj, 'n_hp' : len(params) })),
            ("Sol", SolutionConstructor(hp_map=hp_map)),
            ("DA", DaemonActions()),
            ("Updater", BestSoFarPheromonesUpdater(X=X_train, y=y_train, estimator = DecisionTreeClassifier, bounds=[.1, 5], Q=.05, evaporation_rate = .149)),

        ], 
        iter_max=20,
        id2hp=id2hp, 

    )

    start = perf_counter()
    solutions = aco_pipeline.run(G)['solutions']
    best_sol = max(solutions, key=lambda x: x[1])[0]
    best_params = dict([id2hp[_id] for _id in best_sol[n_feature_obj:]])
    best_cols = best_sol[:n_feature_obj]
    time_aco = perf_counter()-start
    lr_aco = DecisionTreeClassifier(**best_params).fit(X_train[:,best_cols], y_train)
    print(time_aco)


    start = perf_counter()
    search_space = {f'estimator__{k}' : v for k,v in params.items()}
    pipe = Pipeline([
        ('feature_selection', RFE(DecisionTreeClassifier(), n_features_to_select=n_feature_obj) ),
        ('estimator', DecisionTreeClassifier() )
    ])
    clf = GridSearchCV(estimator = pipe, param_grid= search_space)
    clf.fit(X_train, y_train)
    time_classic = perf_counter()-start
    print(f'temps HP : {time_classic}')

    aco_score = cross_val_score(lr_aco,X_test[:,best_cols],y_test, cv=10, scoring=make_scorer(balanced_accuracy_score))
    not_aco_score = cross_val_score(clf.best_estimator_,X_test,y_test, cv=10, scoring=make_scorer(balanced_accuracy_score))
    print(f' ACO : {np.mean(aco_score)=} |  {np.std(aco_score)=} ')
    print(f' Not ACO : {np.mean(not_aco_score)=} |  {np.std(not_aco_score)=}')

    means.append([np.mean(aco_score), np.mean(not_aco_score)])
    variances.append([np.std(aco_score), np.std(not_aco_score)])
    times.append([time_aco, time_classic])

df = pd.DataFrame({'means':means, 'variances' : variances, 'times' : times})

#%%

df.to_csv('trials.csv')

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
datapoints = [[x,y] for x,y in enumerate(G['v'][0,: X.shape[1]])]

datapoints.sort(key = lambda x :x[1])

datapoints = np.array(datapoints)
px.bar(y = datapoints[:,1], x=datapoints[:,0].astype(int).astype(str), category_orders=datapoints[:,0].astype(str), text_auto=True)

# %%
px.imshow((max(rfe.ranking_) - rfe.ranking_.reshape(digits.images[0].shape))/(max(rfe.ranking_)))

# %%
px.imshow(G['v'][0,: X.shape[1]].reshape(digits.images[0].shape))

# %%



X.shape[1]
# %%
