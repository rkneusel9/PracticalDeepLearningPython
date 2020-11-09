#
#  file:  mnist_nn_experiments_base_lr.py
#
#  Reduced MNIST + NN for Chapter 6.
#
#  RTK, 15-Oct-2018
#  Last update:  15-Oct-2018
#
###############################################################

import numpy as np
import time
from sklearn.neural_network import MLPClassifier 


def run(x_train, y_train, x_test, y_test, clf):
    """Train and test"""

    s = time.time()
    clf.fit(x_train, y_train)
    e = time.time()-s
    loss = clf.loss_
    return [clf.score(x_test, y_test), loss, e]


def nn(base_lr, epochs):
    """Initialize a network"""

    return MLPClassifier(solver="sgd", verbose=False, tol=1e-8,
            nesterovs_momentum=False, early_stopping=False,
            learning_rate_init=base_lr, momentum=0.9, max_iter=epochs,
            hidden_layer_sizes=(1000,500), activation="relu",
            learning_rate="constant", batch_size=64)


def main():
    """Run the experiments for the MNIST vector data"""

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  training set samples
    N = 20000
    x = x_train[:N]
    y = y_train[:N]
    xt= x_test[:N]
    yt= y_test[:N]

    base_lr = [0.2,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]

    for lr in base_lr:
        s,l,e = run(x, y, xt, yt, nn(lr,50))
        print("base_lr = %0.5f, score = %0.5f, loss = %0.5f, epochs = %d" % (lr,s,l,50))
    print()

    #  choose epochs so base_lr * epochs == 1.5
    epochs = [8, 15, 30, 150, 300, 1500, 3000, 15000]

    for i in range(len(base_lr)):
        s,l,e = run(x, y, xt, yt, nn(base_lr[i], epochs[i]))
        print("base_lr = %0.5f, score = %0.5f, loss = %0.5f, epochs = %d, time = %0.3f" % (base_lr[i],s,l,epochs[i],e))
    print()

main()

