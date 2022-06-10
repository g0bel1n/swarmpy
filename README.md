# SWARMPY 🐜

Swarmpy is (will be) a Python 3.10 framework to solve some discrete and combinatorial NP-Hard problems in Machine Learning worklows.

First application is on feature selection. The main contribution is to go beyond limited univariate selections. 
One should keep in mind that : 

$$ \min \limits_{ s_{k} \subset \[|0,n|]^{k}} \mathbb{\hat{E}}\[\mathscr{l} (y,f(x_{|s_k}) ] \leq \min_{s_{k}\subset \[|0,n|]^{k}}\sum\limits_{i \in s_k}\mathbb{\hat{E}}\[\mathscr{l}(y,f(x_{|i})] $$

Mainly because of all the non-linearity in the unknown distribution $\mathcal{D}$ from which $x,y$ are sampled ($x,y \in \mathbb{R}^d\times\mathbb{R}$, $k \leq d$)

In plain english : the best subset of features of cardinal k for optimizing the empirical error is not always the subset composed of the k best features for prediction. 

It can be a good approximation and has the merit of avoiding solving this combinatorial problem. Yet it is not optimal. Here, we are proposing to solve the problem using a meta-heuristic stochastic algorithm to find near-optimal solutions.
