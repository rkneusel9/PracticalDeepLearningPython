#
#  file:  mnist_nn_experiments_init.py
#
#  Reduced MNIST + NN for Chapter 11.
#
#  RTK, 20-Oct-2018
#  Last update:  04-Feb-2019
#
###############################################################

import os
import numpy as np
import time
import matplotlib.pylab as plt
from sklearn.neural_network import MLPClassifier 

#
#  Possible weight init methods
#
class Classifier(MLPClassifier):
    """Subclass MLPClassifier to use custom weight initialization"""

    def _init_coef(self, fan_in, fan_out):
        """Custom weight initialization"""

        if (self.init_scheme == 0):
            #  Glorot initialization
            return super(Classifier, self)._init_coef(fan_in, fan_out)
        elif (self.init_scheme == 1):
            #  small uniformly distributed weights
            weights = 0.01*(np.random.random((fan_in, fan_out))-0.5)
            biases = np.zeros(fan_out)
        elif (self.init_scheme == 2):
            #  small Gaussian weights
            weights = 0.005*(np.random.normal(size=(fan_in, fan_out)))
            biases = np.zeros(fan_out)
        elif (self.init_scheme == 3):
            #  He initialization for relu
            weights = np.random.normal(size=(fan_in, fan_out))*  \
                        np.sqrt(2.0/fan_in)
            biases = np.zeros(fan_out)
        elif (self.init_scheme == 4):
            #  Alternate Xavier
            weights = np.random.normal(size=(fan_in, fan_out))*  \
                        np.sqrt(1.0/fan_in)
            biases = np.zeros(fan_out)
#        elif (self.init_scheme == 5):
#            #  small Beta weights
#            weights  = (np.random.beta(2,5, size=(fan_in, fan_out))-0.5)
#            weights += (np.random.beta(5,2, size=(fan_in, fan_out))-0.5)
#            weights *= np.sqrt(1.0/fan_in)
#            biases = np.zeros(fan_out)
#        elif (self.init_scheme == 6):
#            #  small Beta weights
#            weights  = (np.random.beta(2,3, size=(fan_in, fan_out))-0.5)
#            weights += (np.random.beta(3,2, size=(fan_in, fan_out))-0.5)
#            weights *= np.sqrt(1.0/fan_in)
#            biases = np.zeros(fan_out)

        return [weights, biases]


def run(x_train, y_train, x_test, y_test, clf, epochs):
    """Train and test"""

    test_err = []
    clf.max_iter = 1
    for i in range(epochs):
        clf.fit(x_train, y_train)
        terr = 1.0 - clf.score(x_test, y_test)
        clf.warm_start = True
        test_err.append(terr)
    return test_err


def main():
    """Plot the training and validation losses."""

    outdir = "mnist_nn_experiments_init"
    os.system("rm -rf %s" % outdir)
    os.system("mkdir %s" % outdir)

    #  Vector MNIST versions scaled [0,1)
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  Reduce the size of the train dataset
    x_train = x_train[:6000]
    y_train = y_train[:6000]
    epochs = 4000 
    init_types = 5
    trainings = 10

    test_err = np.zeros((trainings, init_types, epochs)) 

    for i in range(trainings):
        for k in range(init_types):
            nn = Classifier(solver="sgd", verbose=False, tol=0,
                   nesterovs_momentum=False, early_stopping=False, learning_rate_init=0.01,
                   momentum=0.9, hidden_layer_sizes=(100,50), activation="relu",
                   alpha=0.2, learning_rate="constant", batch_size=64, max_iter=1)
            nn.init_scheme = k
            test_err[i,k,:] = run(x_train, y_train, x_test, y_test, nn, epochs)

    np.save("mnist_nn_experiments_init_results.npy", test_err)


if (__name__ == "__main__"):
    main()

