import numpy as np
import matplotlib.pylab as plt 
from sklearn import decomposition

x = np.load("../data/iris/iris_features.npy")[:,:2]
y = np.load("../data/iris/iris_labels.npy")
idx = np.where(y != 0)
x = x[idx]
x[:,0] -= x[:,0].mean()
x[:,1] -= x[:,1].mean()

pca = decomposition.PCA(n_components=2)
pca.fit(x)
v = pca.explained_variance_ratio_
   
ax = plt.axes()
plt.scatter(x[:,0],x[:,1],marker='o',color='b')
x0 = v[0]*pca.components_[0,0]
y0 = v[0]*pca.components_[0,1]
ax.arrow(0, 0, x0, y0, head_width=0.05, head_length=0.1, fc='r', ec='r')
x1 = v[1]*pca.components_[1,0]
y1 = v[1]*pca.components_[1,1]
ax.arrow(0, 0, x1, y1, head_width=0.05, head_length=0.1, fc='r', ec='r')
plt.xlabel("$x_0$", fontsize=16)
plt.ylabel("$x_1$", fontsize=16)
plt.show()

