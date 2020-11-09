#
#  file:  mnist_nn_experiments_retrain.py
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


def nn():
    """Initialize a network"""

    return MLPClassifier(solver="sgd", verbose=False, tol=1e-8,
            nesterovs_momentum=False, early_stopping=False,
            learning_rate_init=0.001, momentum=0.9, max_iter=50,
            hidden_layer_sizes=(1000,500), activation="relu",
            batch_size=64)


def main():
    """Run the experiments for the iris data"""

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

    M = 20
    scores = np.zeros(M)
    losses = np.zeros(M)
    for i in range(M):
        s,l,e = run(x, y, xt, yt, nn())
        print("%03i: score = %0.5f, loss = %0.5f" % (i,s,l))
        scores[i] = s
        losses[i] = l

    print()
    print("Scores:  min, max, mean+/-SE: %0.5f, %0.5f, %0.5f +/- %0.5f" % \
        (scores.min(), scores.max(), scores.mean(), scores.std()/np.sqrt(scores.shape[0])))
    print("Loss  :  min, max, mean+/-SE: %0.5f, %0.5f, %0.5f +/- %0.5f" % \
        (losses.min(), losses.max(), losses.mean(), losses.std()/np.sqrt(losses.shape[0])))
    print()


main()

