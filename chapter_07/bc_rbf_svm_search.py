import numpy as np
from sklearn.svm import SVC 

def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)

def split(x,y,k,m):
    ns = int(y.shape[0]/m)
    s = []
    for i in range(m):
        s.append([x[(ns*i):(ns*i+ns)], y[(ns*i):(ns*i+ns)]])
    x_test, y_test = s[k]
    x_train = []
    y_train = []
    for i in range(m):
        if (i==k):
            continue
        else:
            a,b = s[i]
            x_train.append(a)
            y_train.append(b)
    x_train = np.array(x_train).reshape(((m-1)*ns,30))
    y_train = np.array(y_train).reshape((m-1)*ns)
    return [x_train, y_train, x_test, y_test]

def main():
    m = 5 
    x = np.load("../data/breast/bc_features_standard.npy")
    y = np.load("../data/breast/bc_labels.npy")
    idx = np.argsort(np.random.random(y.shape[0]))
    x = x[idx]
    y = y[idx]

    Cs = np.array([0.01,0.1,1.0,2.0,10.0,50.0,100.0])
    gs = (1./30)*2.0**np.array([-4,-3,-2,-1,0,1,2,3])
    zmax = 0.0 
    for C in Cs: 
        for g in gs: 
            z = np.zeros(m)
            for k in range(m):
                x_train, y_train, x_test, y_test = split(x,y,k,m)
                z[k] = run(x_train, y_train, x_test, y_test, SVC(C=C,gamma=g,kernel="rbf"))
            if (z.mean() > zmax):
                zmax = z.mean()
                bestC = C 
                bestg = g 
    print("best C     = %0.5f" % bestC)
    print("     gamma = %0.5f" % bestg)
    print("   accuracy= %0.5f" % zmax)

main()

