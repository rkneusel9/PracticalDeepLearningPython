#
#  file:  mnist_nn_experiments_batch_size.py
#
#  RTK, 14-Oct-2018
#  Last update:  07-Jan-2019
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
    weights = clf.coefs_
    biases = clf.intercepts_
    params = 0
    for w in weights:
        params += w.shape[0]*w.shape[1]
    for b in biases:
        params += b.shape[0]
    return [clf.score(x_test, y_test), loss, params, e, clf.n_iter_]


def nn(bz,epochs):
    """Initialize a network"""

    return MLPClassifier(solver="sgd", verbose=False, tol=1e-8,
            nesterovs_momentum=False, early_stopping=False,
            learning_rate_init=0.001, momentum=0.9, max_iter=epochs,
            hidden_layer_sizes=(1000,500), activation="relu",
            batch_size=bz)


def main():
    """Run the experiments for the iris data"""

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  training set samples
    N = 16384
    x = x_train[:N]
    y = y_train[:N]

    batch_sizes = [16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4,2]
    M = 8192  # set epochs so minibatches is constant

    for bz in batch_sizes:
        print("batch size = %4d:" % bz)
        #epochs = 100 
        epochs = (M*bz) // N
        if (epochs < 1):
            epochs = 1
        scores = []
        loss = []
        tm = []
        for i in range(5):
            s,l,p,e,m = run(x, y, x_test, y_test, nn(bz,epochs))
            scores.append(s)
            loss.append(l)
            tm.append(e)
            print("    score = %0.5f, loss = %0.5f, epochs = %d, actual = %d" % (s,l,epochs,m))
        scores = np.array(scores)
        loss = np.array(loss)
        sm = scores.mean()
        se = scores.std() / np.sqrt(scores.shape[0])
        lm = loss.mean()
        le = loss.std() / np.sqrt(loss.shape[0])
        print("    final score = %0.5f +/- %0.5f, loss = %0.5f +/- %0.5f, epochs = %d" % (sm,se,lm,le,epochs))


main()

