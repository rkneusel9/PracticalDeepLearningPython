#
#  file:  mnist_nn_experiments_samples.py
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


def nn(epochs):
    """Initialize a network"""

    return MLPClassifier(solver="sgd", verbose=False, tol=1e-8,
            nesterovs_momentum=False, early_stopping=False,
            learning_rate_init=0.05, momentum=0.9, max_iter=epochs,
            hidden_layer_sizes=(1000,500), activation="relu",
            learning_rate="constant", batch_size=100)


def main():
    """Run the experiments for the MNIST vector data"""

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  training set samples
    N = [100, 200, 300, 400, 500, 600,
        700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000,
        4500, 5000, 7500, 10000, 15000, 20000, 25000, 30000]
    M = 5

    for n in N:
        scores = np.zeros(M)
        print("samples = %5d" % n)
        for i in range(M):
            idx = np.argsort(np.random.random(y_train.shape[0]))
            x_train = x_train[idx]
            y_train = y_train[idx]
            x = x_train[:n]
            y = y_train[:n]
            epochs = int((100.0/n)*1000) # epochs to take 1,000 SGD steps
            s,l,e = run(x, y, x_test, y_test, nn(epochs))
            scores[i] = s
            print("    score = %0.5f, loss = %0.5f, epochs = %d, training time = %0.3f" % (s,l,epochs,e))
        print("    mean score = %0.5f +/- %0.5f" % (scores.mean(), scores.std()/np.sqrt(M)))
        print()
    print()


main()

