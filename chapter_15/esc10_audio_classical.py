#
#  file:  esc10_audio_classical.py
#
#  Apply classical models to the ESC-10 raw audio dataset.
#
#  RTK, 13-Nov-2019
#  Last update:  13-Nov-2019
#
###############################################################

import time
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

def run(x_train, y_train, x_test, y_test, clf):
    """Train and test"""

    s = time.time()
    clf.fit(x_train, y_train)
    e_train = time.time() - s
    s = time.time()
    score = clf.score(x_test, y_test)
    e_test = time.time() - s
    print("score = %0.4f (time, train=%8.3f, test=%8.3f)" % (score, e_train, e_test))


def train(x_train, y_train, x_test, y_test):
    """Train the models on the given dataset"""

    print("    Nearest centroid          : ", end='')
    run(x_train, y_train, x_test, y_test, NearestCentroid())
    print("    k-NN classifier (k=3)     : ", end='')
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
    print("    k-NN classifier (k=7)     : ", end='')
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))
    print("    Naive Bayes (Gaussian)    : ", end='')
    run(x_train, y_train, x_test, y_test, GaussianNB())
    print("    Random Forest (trees=  5) : ", end='')
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=5))
    print("    Random Forest (trees= 50) : ", end='')
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=50))
    print("    Random Forest (trees=500) : ", end='')
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=500))
    print("    Random Forest (trees=1000): ", end='')
    run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=1000))
    print("    LinearSVM (C=0.01)        : ", end='')
    run(x_train, y_train, x_test, y_test, LinearSVC(C=0.01))
    print("    LinearSVM (C=0.1)         : ", end='')
    run(x_train, y_train, x_test, y_test, LinearSVC(C=0.1))
    print("    LinearSVM (C=1.0)         : ", end='')
    run(x_train, y_train, x_test, y_test, LinearSVC(C=1.0))
    print("    LinearSVM (C=10.0)        : ", end='')
    run(x_train, y_train, x_test, y_test, LinearSVC(C=10.0))


def main():
    """Run the experiments for the ESC-10 data"""

    #  Load the data and scale
    x_train = np.load("../data/audio/ESC-10/esc10_raw_train_audio.npy")[:,:,0]
    y_train = np.load("../data/audio/ESC-10/esc10_raw_train_labels.npy")
    x_test  = np.load("../data/audio/ESC-10/esc10_raw_test_audio.npy")[:,:,0]
    y_test  = np.load("../data/audio/ESC-10/esc10_raw_test_labels.npy")

    x_train = (x_train.astype('float32') + 32768) / 65536
    x_test = (x_test.astype('float32') + 32768) / 65536

    #  Train and test the models
    train(x_train, y_train, x_test, y_test)


main()

