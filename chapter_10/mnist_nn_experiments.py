import numpy as np
import time
from sklearn.neural_network import MLPClassifier

def run(x_train, y_train, x_test, y_test, clf):
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
    return MLPClassifier(solver="sgd", verbose=False, tol=1e-8,
            nesterovs_momentum=False, early_stopping=False,
            learning_rate_init=0.001, momentum=0.9, max_iter=200,
            hidden_layer_sizes=layers, activation=act)

def main():
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")/256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")/256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    N = 1000
    x_train = x_train[:N]
    y_train = y_train[:N]
    x_test  = x_test[:N]
    y_test  = y_test[:N]

    layers = [
        (1,), (500,), (800,), (1000,), (2000,), (3000,),
        (1000,500), (3000,1500),
        (2,2,2), (1000,500,250), (2000,1000,500),
    ]

    for act in ["relu", "logistic", "tanh"]:
        print("%s:" % act)
        for layer in layers:
            scores = []
            loss = []
            tm = []
            for i in range(10):
                s,l,params,e = run(x_train, y_train, x_test, y_test, nn(layer,act))
                scores.append(s)
                loss.append(l)
                tm.append(e)
            s = np.array(scores)
            l = np.array(loss)
            t = np.array(tm)
            n = np.sqrt(s.shape[0])
            print("    layers: %14s, score= %0.4f +/- %0.4f, loss = %0.4f +/- %0.4f (params = %6d, time = %0.2f s)" % \
                (str(layer), s.mean(), s.std()/n, l.mean(), l.std()/n, params, t.mean()))

main()

