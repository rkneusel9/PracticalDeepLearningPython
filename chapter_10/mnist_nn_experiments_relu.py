#
#  file:  mnist_nn_experiments_relu.py
#
#  Reduced MNIST + NN for Chapter 6.
#
#  RTK, 13-Oct-2018
#  Last update:  30-Dec-2018
#
###############################################################

import numpy as np
import time
from sklearn.neural_network import MLPClassifier 

def nparams(x_train, y_train, clf):
    clf.max_iter=1
    clf.fit(x_train, y_train)
    weights = clf.coefs_
    biases = clf.intercepts_
    params = 0
    for w in weights:
        params += w.shape[0]*w.shape[1]
    for b in biases:
        params += b.shape[0]
    return params


def run(x_train, y_train, x_test, y_test, clf):
    """Train and test"""

    s = time.time()
    clf.fit(x_train, y_train)
    e = time.time()-s
    loss = clf.loss_
    weights = clf.coefs_
    biases = clf.intercepts_
    params = 0
    for w in weights:
        params += w.shape[0]*w.shape[1]
    for b in biases:
        params += b.shape[0]
    return [clf.score(x_test, y_test), loss, params, e]


def nn(layers, act):
    """Initialize a network"""

    return MLPClassifier(solver="sgd", verbose=False, tol=1e-8,
            nesterovs_momentum=False, early_stopping=False,
            learning_rate_init=0.001, momentum=0.9, max_iter=200,
            hidden_layer_sizes=layers, activation=act)


def main():
    """Run the experiments for the MNIST data"""

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  Reduce the size of the train dataset
    N = 20000
    x_train = x_train[:N]
    y_train = y_train[:N]
    x_test  = x_test[:N]
    y_test  = y_test[:N]

    #  chosen so # params approx same across respective number of layers
    layers = [
        (1000,), (2000,), (4000,), (8000,),
        (700,350), (1150,575), (1850,925), (2850,1425),
        (660, 330, 165), (1080,540,270), (1714,857,429), (2620,1310,655),
    ]

    layers = [(8000,),(2850,1425)]

    for layer in layers:
        scores = []
        loss = []
        tm = []
        for i in range(5):
            s,l,params,e = run(x_train, y_train, x_test, y_test, nn(layer,"relu"))
            scores.append(s)
            loss.append(l)
            tm.append(e)
        s = np.array(scores)
        l = np.array(loss)
        t = np.array(tm)
        n = np.sqrt(s.shape[0])
        print("layers: %14s, score= %0.4f +/- %0.4f, loss = %0.4f +/- %0.4f (params = %6d, time = %0.2f s)" % \
            (str(layer), s.mean(), s.std()/n, l.mean(), l.std()/n, params, t.mean()))


main()

