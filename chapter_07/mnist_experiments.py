import time
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import decomposition

def run(x_train, y_train, x_test, y_test, clf):
    s = time.time()
    clf.fit(x_train, y_train)
    e_train = time.time() - s 
    s = time.time()
    score = clf.score(x_test, y_test)
    e_test = time.time() - s 
    print("score = %0.4f (time, train=%8.3f, test=%8.3f)" % (score, e_train, e_test))

def train(x_train, y_train, x_test, y_test):
    print("    Nearest centroid          : ", end='')
    run(x_train, y_train, x_test, y_test, NearestCentroid())
    print("    k-NN classifier (k=3)     : ", end='')
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
    print("    k-NN classifier (k=7)     : ", end='')
    run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))
    print("    Naive Bayes (Gaussian)    : ", end='')
    run(x_train, y_train, x_test, y_test, GaussianNB())
    print("    Decision Tree             : ", end='')
    run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())
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
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")
    y_test = np.load("../data/mnist/mnist_test_labels.npy")

    print("Models trained on raw [0,255] images:")
    train(x_train, y_train, x_test, y_test)
    print("Models trained on raw [0,1) images:")
    train(x_train/256.0, y_train, x_test/256.0, y_test)

    m = x_train.mean(axis=0)
    s = x_train.std(axis=0) + 1e-8
    x_ntrain = (x_train - m) / s
    x_ntest  = (x_test - m) / s

    print("Models trained on normalized images:")
    train(x_ntrain, y_train, x_ntest, y_test)

    pca = decomposition.PCA(n_components=15)
    pca.fit(x_ntrain)
    x_ptrain = pca.transform(x_ntrain)
    x_ptest = pca.transform(x_ntest)
    
    print("Models trained on first 15 PCA components of normalized images:")
    train(x_ptrain, y_train, x_ptest, y_test)

main()

