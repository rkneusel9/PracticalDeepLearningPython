#
#  file:  mnist_nn_experiments_momentum.py
#
#  RTK, 19-Oct-2018
#  Last update:  03-Feb-2019
#
###############################################################

import os
import time
import numpy as np
import matplotlib.pylab as plt
from sklearn.neural_network import MLPClassifier


def epoch(x_train, y_train, x_test, y_test, clf):
    """Results for a single epoch"""

    clf.fit(x_train, y_train)
    train_loss = clf.loss_
    train_err = 1.0 - clf.score(x_train, y_train)
    val_err = 1.0 - clf.score(x_test, y_test)
    clf.warm_start = True
    return [train_loss, train_err, val_err]


def run(x_train, y_train, x_test, y_test, clf, max_iter):
    """Train and test"""

    train_loss = []
    train_err = []
    val_err = []

    clf.max_iter = 1  # one epoch at a time
    for i in range(max_iter):
        tl, terr, verr = epoch(x_train, y_train, x_test, y_test, clf)
        train_loss.append(tl)
        train_err.append(terr)
        val_err.append(verr)
        print("    %4d: val_err = %0.5f" % (i, val_err[-1]))

    return [train_loss, train_err, val_err]


def main():
    """Plot the training and validation losses."""

    os.system("rm -rf mnist_nn_experiments_momentum")
    os.system("mkdir mnist_nn_experiments_momentum")

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  Reduce the size of the train dataset
    x_train = x_train[:3000]
    y_train = y_train[:3000]

    #  momentum values
    colors= ['k','r','b','g','c','m']
    momentum = [0.0,0.3,0.5,0.7,0.9,0.99]
    epochs = 10000 

    for k,m in enumerate(momentum):
        nn = MLPClassifier(solver="sgd", verbose=False, tol=0,
                nesterovs_momentum=False,
                early_stopping=False,
                learning_rate_init=0.01,
                momentum=m,
                hidden_layer_sizes=(100,50),
                activation="relu",
                alpha=0.0001,
                learning_rate="constant",
                batch_size=64,
                max_iter=1)
        print("momentum = %0.1f" % m)
        train_loss, train_err, val_err = run(x_train, y_train, x_test, y_test, nn, epochs)
        print("    final: train error: %0.5f, val error: %0.5f"  % \
            (train_err[-1], val_err[-1]))
        print()
        if (k==0):
            plt.plot(val_err, color=colors[k], linewidth=3)
        else:
            plt.plot(val_err, color=colors[k])
        np.save("mnist_nn_experiments_momentum/train_error_%0.2f.npy" % m, train_err)
        np.save("mnist_nn_experiments_momentum/train_loss_%0.2f.npy" % m, train_loss)
        np.save("mnist_nn_experiments_momentum/val_error_%0.2f.npy" % m, val_err)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.tight_layout()
    pname = "mnist_nn_experiments_momentum/mnist_nn_experiments_momentum_plot.png"
    plt.savefig(pname, format="png", dpi=600)
    plt.close()


if (__name__ == "__main__"):
    main()

