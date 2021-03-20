#
#  file:  mnist_2x2_tables.py
#
#  Different MNIST dataset models for Chapter 12.
#
#  RTK, 04-Apr-2019
#  Last update:  08-Apr-2019
#
###############################################################

import time
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import decomposition

def tally_predictions(clf, x, y): 
    p = clf.predict(x)
    score = clf.score(x,y)
    tp = tn = fp = fn = 0 
    for i in range(len(y)):
        if (p[i] == 0) and (y[i] == 0): 
            tn += 1
        elif (p[i] == 0) and (y[i] == 1): 
            fn += 1
        elif (p[i] == 1) and (y[i] == 0): 
            fp += 1
        else:
            tp += 1
    return [tp, tn, fp, fn, score]


def basic_metrics(tally):
    """Use the tallies to generate basic metrics"""

    tp, tn, fp, fn, _ = tally
    return {
        "TPR": tp / (tp + fn),
        "TNR": tn / (tn + fp),
        "PPV": tp / (tp + fp),
        "NPV": tn / (tn + fn),
        "FPR": fp / (fp + tn),
        "FNR": fn / (fn + tp)
    }


from math import sqrt
def advanced_metrics(tally, m):
    """Use the tallies to calculate more advanced metrics"""

    tp, tn, fp, fn, _ = tally
    n = tp+tn+fp+fn

    po = (tp+tn)/n
    pe = (tp+fn)*(tp+fp)/n**2 + (tn+fp)*(tn+fn)/n**2

    return {
        "F1": 2.0*m["PPV"]*m["TPR"] / (m["PPV"] + m["TPR"]),
        "MCC": (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        "kappa": (po - pe) / (1.0 - pe),
        "informedness": m["TPR"] + m["TNR"] - 1.0,
        "markedness": m["PPV"] + m["NPV"] - 1.0
    }


def pp(t,m,b):
    """Print the metrics"""

    tp, tn, fp, fn, score = t
    print("    TP=%5d  FP=%5d" % (tp,fp))
    print("    FN=%5d  TN=%5d, score=%0.4f" % (fn,tn, score))
    print()
    print("    TPR=%0.4f  TNR=%0.4f" % (m["TPR"], m["TNR"]))
    print("    PPV=%0.4f  NPV=%0.4f" % (m["PPV"], m["NPV"]))
    print("    FPR=%0.4f  FNR=%0.4f" % (m["FPR"], m["FNR"]))
    print()
    print("    F1          = %0.4f" % b["F1"])
    print("    MCC         = %0.4f" % b["MCC"])
    print("    kappa       = %0.4f" % b["kappa"])
    print("    informedness= %0.4f" % b["informedness"])
    print("    markedness  = %0.4f" % b["markedness"])
    print()


def run(x_train, y_train, x_test, y_test, clf):
    """Train and test"""

    s = time.time()
    clf.fit(x_train, y_train)
    tallies = tally_predictions(clf, x_test, y_test)
    bmetrics = basic_metrics(tallies)
    ametrics = advanced_metrics(tallies, bmetrics)
    pp(tallies, bmetrics, ametrics)


def train(x_train, y_train, x_test, y_test):
    """Train the models on the given dataset"""

    print("Nearest centroid          : ")
    run(x_train, y_train, x_test, y_test, NearestCentroid())
    print("k-NN classifier (k=3)     : ")
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
    print("k-NN classifier (k=7)     : ")
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))
    print("Naive Bayes (Gaussian)    : ")
    run(x_train, y_train, x_test, y_test, GaussianNB())
    print("Decision Tree             : ")
    run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())
    print("Random Forest (trees=  5) : ")
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=5))
    print("Random Forest (trees= 50) : ")
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=50))
    print("Random Forest (trees=500) : ")
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=500))
    print("Random Forest (trees=1000): ")
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=1000))
    print("LinearSVM (C=0.01)        : ")
    run(x_train, y_train, x_test, y_test, LinearSVC(C=0.01))
    print("LinearSVM (C=0.1)         : ")
    run(x_train, y_train, x_test, y_test, LinearSVC(C=0.1))
    print("LinearSVM (C=1.0)         : ")
    run(x_train, y_train, x_test, y_test, LinearSVC(C=1.0))
    print("LinearSVM (C=10.0)        : ")
    run(x_train, y_train, x_test, y_test, LinearSVC(C=10.0))


def main():
    """Run the experiments for the MNIST data"""

    #  Load the data and normalize
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64") / 256.0
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64") / 256.0
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    #  Keep only 3 and 5
    i = np.where(y_train == 3)[0]
    xtrn3 = x_train[i]
    i = np.where(y_train == 5)[0]
    xtrn5 = x_train[i]
    i = np.where(y_test == 3)[0]
    xtst3 = x_test[i]
    i = np.where(y_test == 5)[0]
    xtst5 = x_test[i]
    xtrn = np.concatenate((xtrn3,xtrn5))
    xtst = np.concatenate((xtst3,xtst5))
    ytrn = np.zeros(len(xtrn3)+len(xtrn5))
    ytrn[:len(xtrn3)] = 0
    ytrn[len(xtrn3):] = 1
    ytst = np.zeros(len(xtst3)+len(xtst5))
    ytst[:len(xtst3)] = 0
    ytst[len(xtst3):] = 1

    np.random.seed(12345)  #  make reproducible
    i = np.argsort(np.random.random(size=len(ytrn)))
    xtrn = xtrn[i]
    ytrn = ytrn[i]
    i = np.argsort(np.random.random(size=len(ytst)))
    xtst = xtst[i]
    ytst = ytst[i]

    train(xtrn, ytrn, xtst, ytst)


main()

