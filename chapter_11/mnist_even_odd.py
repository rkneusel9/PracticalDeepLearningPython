#
#  file:  mnist_even_odd.py
#
#  Reduced MNIST + NN for Chapter 12.
#
#  RTK, 12-Apr-2019
#  Last update:  12-Apr-2019
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
    score = clf.score(x_test, y_test)
    prob = clf.predict_proba(x_test)

    return [score, loss, prob, e]


def nn(layers):
    """Initialize a network"""

    return MLPClassifier(solver="sgd", verbose=False, tol=1e-8,
            nesterovs_momentum=False, early_stopping=False,
            batch_size=64,
            learning_rate_init=0.001, momentum=0.9, max_iter=200,
            hidden_layer_sizes=layers, activation="relu")


def main():
    """Run the experiments for the MNIST data"""

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_even_odd_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_even_odd_labels.npy")

    #  Reduce the size of the train dataset
    N = 1000
    x_train = x_train[:N]
    y_train = y_train[:N]

    layers = [(2,), (100,), (100,50), (500,250)]
    mlayers = ["2", "100", "100x50", "500x250"]

    for i,layer in enumerate(layers):
        score,loss,prob,tm = run(x_train, y_train, x_test, y_test, nn(layer))
        print("layers: %s, score= %0.6f, loss = %0.6f (time = %0.2f s)" % \
            (mlayers[i], score, loss, tm))
        np.save("mnist_even_odd_probs_%s.npy" % mlayers[i], prob)
    print()


main()

