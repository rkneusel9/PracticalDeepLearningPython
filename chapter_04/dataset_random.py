import numpy as np
from sklearn.datasets import make_classification

x,y = make_classification(n_samples=10000, weights=(0.9,0.1))
idx = np.argsort(np.random.random(y.shape[0]))
x = x[idx]
y = y[idx]

ntrn = int(0.9*y.shape[0])
nval = int(0.05*y.shape[0])

xtrn = x[:ntrn]
ytrn = y[:ntrn]
xval = x[ntrn:(ntrn+nval)]
yval = y[ntrn:(ntrn+nval)]
xtst = x[(ntrn+nval):]
ytst = y[(ntrn+nval):]

