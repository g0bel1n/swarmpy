#%%
import numpy as np
from swarmpy_tsp import (
    DaemonActions,
    SolutionConstructor,
    Planner,
    BestTourPheromonesUpdater,
    BestSoFarPheromonesUpdater,
    ProportionnalPheromonesUpdater,
    ACO_Iterator,
    Antcoder,
)

import plotly.express as px


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


#%%

G, opt_score = Antcoder()
G1, _ = Antcoder()

G["e"] *= 0.5
G1["e"] *= 0.5
#%%
p_Daemon = ACO_Iterator(
    [
        ("Planner", Planner({"alpha": 1.0, "beta": 2.0})),
        ("Sol", SolutionConstructor()),
        ("DA", DaemonActions()),
        ("Updater", BestTourPheromonesUpdater(bounds=[0.1, 1])),
    ]
)

p = ACO_Iterator(
    [
        ("Planner", Planner({"alpha": 1.0, "beta": 2.0})),
        ("Sol", SolutionConstructor()),
        ("Updater", BestTourPheromonesUpdater(bounds=[0.1, 1])),
    ]
)
#%%
sols_DA = p_Daemon.run(G=G, iter_max=100)
sols = p.run(G=G1, iter_max=100)

#%%
scores_DA = np.array([el[1] for el in sols_DA]) - opt_score
scores = np.array([el[1] for el in sols]) - opt_score

# %%
px.line(y=[scores_DA, scores], x=np.arange(scores.shape[0]))

# %%
opt_score

# %%
print(len(sols_DA[0][0]))

# %%
