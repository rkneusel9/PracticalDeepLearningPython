import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
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
    return [score, e_train, e_test]

def main():
    x_train = np.load("../data/mnist/mnist_train_vectors.npy").astype("float64")
    y_train = np.load("../data/mnist/mnist_train_labels.npy")
    x_test = np.load("../data/mnist/mnist_test_vectors.npy").astype("float64")
    y_test = np.load("../data/mnist/mnist_test_labels.npy")
    m = x_train.mean(axis=0)
    s = x_train.std(axis=0) + 1e-8
    x_ntrain = (x_train - m) / s 
    x_ntest  = (x_test - m) / s 

    n = 78
    pcomp = np.linspace(10,780,n, dtype="int16")
    nb=np.zeros((n,4))
    rf=np.zeros((n,4))
    sv=np.zeros((n,4))
    tv=np.zeros((n,2))

    for i,p in enumerate(pcomp):
        pca = decomposition.PCA(n_components=p)
        pca.fit(x_ntrain)
        xtrain = pca.transform(x_ntrain)
        xtest = pca.transform(x_ntest)
        tv[i,:] = [p, pca.explained_variance_ratio_.sum()]
        sc,etrn,etst =run(xtrain, y_train, xtest, y_test, GaussianNB())
        nb[i,:] = [p,sc,etrn,etst]
        sc,etrn,etst =run(xtrain, y_train, xtest, y_test, RandomForestClassifier(n_estimators=50))
        rf[i,:] = [p,sc,etrn,etst]
        sc,etrn,etst =run(xtrain, y_train, xtest, y_test, LinearSVC(C=1.0))
        sv[i,:] = [p,sc,etrn,etst]

    np.save("../data/mnist/mnist_pca_tv.npy", tv) 
    np.save("../data/mnist/mnist_pca_nb.npy", nb)
    np.save("../data/mnist/mnist_pca_rf.npy", rf)
    np.save("../data/mnist/mnist_pca_sv.npy", sv)

main()

