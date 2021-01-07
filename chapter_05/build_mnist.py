import numpy as np
import keras
from keras.datasets import mnist

(xtrn, ytrn), (xtst, ytst) = mnist.load_data()
idx = np.argsort(np.random.random(ytrn.shape[0]))
xtrn = xtrn[idx]
ytrn = ytrn[idx]
idx = np.argsort(np.random.random(ytst.shape[0]))
xtst = xtst[idx]
ytst = ytst[idx]

np.save("../data/mnist/mnist_train_images.npy", xtrn)
np.save("../data/mnist/mnist_train_labels.npy", ytrn)
np.save("../data/mnist/mnist_test_images.npy", xtst)
np.save("../data/mnist/mnist_test_labels.npy", ytst)

xtrnv = xtrn.reshape((60000,28*28))
xtstv = xtst.reshape((10000,28*28))
np.save("../data/mnist/mnist_train_vectors.npy", xtrnv)
np.save("../data/mnist/mnist_test_vectors.npy", xtstv)

idx = np.argsort(np.random.random(28*28))
for i in range(60000):
    xtrnv[i,:] = xtrnv[i,idx]
for i in range(10000):
    xtstv[i,:] = xtstv[i,idx]
np.save("../data/mnist/mnist_train_scrambled_vectors.npy", xtrnv)
np.save("../data/mnist/mnist_test_scrambled_vectors.npy", xtstv)

t = np.zeros((60000,28,28))
for i in range(60000):
    t[i,:,:] = xtrnv[i,:].reshape((28,28))
np.save("../data/mnist/mnist_train_scrambled_images.npy", t)
t = np.zeros((10000,28,28))
for i in range(10000):
    t[i,:,:] = xtstv[i,:].reshape((28,28))
np.save("../data/mnist/mnist_test_scrambled_images.npy", t)

