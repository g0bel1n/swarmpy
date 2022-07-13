#%%

import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import itertools

#%%

X = np.random.uniform(size=1000).reshape((100,10))
# %%
px.scatter(X)
# %%
params  = {'penalty' : ['l1','l2', 'none', 'elasticnet'], 'C' : [i/10 for i in range(10)], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
#%%


params_lens = [len(v) for v in params.values()]

total_len = np.sum(params_lens)
#%%
phero_matrix = np.ones((X.shape[1]+total_len, X.shape[1]+total_len))
# %%
params_lens = [0] + list(np.cumsum(params_lens))
for n1, n2 in itertools.pairwise(params_lens) :
    phero_matrix[X.shape[1]+n1:X.shape[1]+n2,X.shape[1]+n1:X.shape[1]+n2] = 0

# %%

