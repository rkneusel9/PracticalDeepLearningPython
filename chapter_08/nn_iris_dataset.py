import numpy as np
d = np.load("../data/iris/iris_train_features_augmented.npy")
l = np.load("../data/iris/iris_train_labels_augmented.npy")
d1 = d[np.where(l==1)]
d2 = d[np.where(l==2)]
a=len(d1)
b=len(d2)
x = np.zeros((a+b,2))
x[:a,:] = d1[:,2:]
x[a:,:] = d2[:,2:]
y = np.array([0]*a+[1]*b)
i = np.argsort(np.random.random(a+b))
x = x[i]
y = y[i]
np.save("../data/iris/iris2_train.npy", x)
np.save("../data/iris/iris2_train_labels.npy", y)
d = np.load("../data/iris/iris_test_features_augmented.npy")
l = np.load("../data/iris/iris_test_labels_augmented.npy")
d1 = d[np.where(l==1)]
d2 = d[np.where(l==2)]
a=len(d1)
b=len(d2)
x = np.zeros((a+b,2))
x[:a,:] = d1[:,2:]
x[a:,:] = d2[:,2:]
y = np.array([0]*a+[1]*b)
i = np.argsort(np.random.random(a+b))
x = x[i]
y = y[i]
np.save("../data/iris/iris2_test.npy", x)
np.save("../data/iris/iris2_test_labels.npy", y)

