def confusion_matrix(y_test, y_predict, n=10):
    cmat = np.zeros((n,n), dtype="uint32")
    for i,y in enumerate(y_test):
        cmat[y, y_predict[i]] += 1
    return cmat


def weighted_mean_acc(cmat):
    N = cmat.sum()
    C = cmat.sum(axis=1)
    return ((C/N)*(100*np.diag(cmat)/C)).sum()

