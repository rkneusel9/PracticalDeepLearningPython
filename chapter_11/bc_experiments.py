#
#  file:  bc_experiments.py
#
#  Different breast cancer dataset models for Chapter 4.
#
#  RTK, 02-Jun-2018
#  Last update:  02-Jun-2018
#
###############################################################

import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def run(x_train, y_train, x_test, y_test, clf):
    """Train and test"""

    clf.fit(x_train, y_train)
    print("    score = %0.4f" % clf.score(x_test, y_test))
    print()


def main():
    """Run the experiments for the iris data"""

    #  Load the original data and build a train/test split
    x = np.load("../data/breast/bc_features_standard.npy")
    y = np.load("../data/breast/bc_labels.npy")

    np.random.seed(12345)
    idx = np.argsort(np.random.random(y.shape[0]))
    x = x[idx]
    y = y[idx]
    np.random.seed()

    N = 455
    x_train = x[:N];  x_test = x[N:]
    y_train = y[:N];  y_test = y[N:]

    #  Nearest centroid
    print("Nearest centroid:")
    run(x_train, y_train, x_test, y_test, NearestCentroid())

    #  k-NN
    print("k-NN classifier (k=3):")
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
    print("k-NN classifier (k=7):")
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))

    #  Naive Bayes
    print("Naive Bayes classifier (Gaussian):")
    run(x_train, y_train, x_test, y_test, GaussianNB())

    #  Decision tree
    print("Decision Tree classifier:")
    run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())

    #  Random forest
    print("Random Forest classifier (estimators=5):")
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=5))
    print("Random Forest classifier (estimators=50):")
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=50))

    #  SVM - linear
    print("SVM (linear, C=1.0):")
    run(x_train, y_train, x_test, y_test, SVC(kernel="linear", C=1.0))

    #  SVM - RBF - gamma = 1/30 = 0.03333
    print("SVM (RBF, C=1.0, gamma=0.03333):")
    run(x_train, y_train, x_test, y_test, SVC(kernel="rbf", C=1.0, gamma=0.03333))


main()

