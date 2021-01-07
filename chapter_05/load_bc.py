import numpy as np
import matplotlib.pyplot as plt

with open("../data/breast/wdbc.data") as f:
    lines = [i[:-1] for i in f.readlines() if i != ""] 

n = ["B","M"]
x = np.array([n.index(i.split(",")[1]) for i in lines],dtype="uint8")
y = np.array([[float(j) for j in i.split(",")[2:]] for i in lines])
i = np.argsort(np.random.random(x.shape[0]))
x = x[i]
y = y[i]
z = (y - y.mean(axis=0)) / y.std(axis=0)

np.save("../data/breast/bc_features.npy", y)
np.save("../data/breast/bc_features_standard.npy", z)
np.save("../data/breast/bc_labels.npy", x)
plt.boxplot(z)
plt.show()

