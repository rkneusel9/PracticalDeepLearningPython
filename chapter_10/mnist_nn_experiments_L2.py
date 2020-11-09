#
#  file:  mnist_nn_experiments_L2.py
#
#  RTK, 19-Oct-2018
#  Last update:  20-Oct-2018
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

    wavg = 0.0
    n = 0
    for w in clf.coefs_:
        wavg += w.sum()
        n += w.size
    wavg /= n

    return [train_loss, train_err, val_err, wavg]


def main():
    """Plot the training and validation losses."""

    os.system("rm -rf mnist_nn_experiments_L2")
    os.system("mkdir mnist_nn_experiments_L2")

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  Reduce the size of the train dataset
    x_train = x_train[:3000]
    y_train = y_train[:3000]

    #  L2 values
    colors= ['k','r','b','g','c']
    alpha = [0.0,0.1,0.2,0.3,0.4]
    epochs = 10000 

    for k,a in enumerate(alpha):
        nn = MLPClassifier(solver="sgd", verbose=False, tol=0,
                nesterovs_momentum=False,
                early_stopping=False,
                learning_rate_init=0.01,
                momentum=0.0,
                hidden_layer_sizes=(100,50),
                activation="relu",
                alpha=a,
                learning_rate="constant",
                batch_size=64,
                max_iter=1)
        tt = "alpha = %0.6f" % a
        print(tt)
        train_loss, train_err, val_err, wavg = run(x_train, y_train, x_test, y_test, nn, epochs)
        print("    final: train error: %0.5f, val error: %0.5f, mean weight value = %0.8f"  % \
            (train_err[-1], val_err[-1], wavg))
        print()
        if (k==0):
            plt.plot(val_err, color=colors[k], linewidth=3)
        else:
            plt.plot(val_err, color=colors[k])
        np.save("mnist_nn_experiments_L2/train_error_%0.6f.npy" % a, train_err)
        np.save("mnist_nn_experiments_L2/train_loss_%0.6f.npy" % a, train_loss)
        np.save("mnist_nn_experiments_L2/val_error_%0.6f.npy" % a, val_err)
        np.save("mnist_nn_experiments_L2/mean_weight_%0.6f.npy" % a, np.array(wavg))
    plt.ylim((0.03,0.17))
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    pname = "mnist_nn_experiments_L2/mnist_nn_experiments_L2_plot.png"
    plt.savefig(pname, format="png", dpi=600)
    plt.close()


if (__name__ == "__main__"):
    main()

