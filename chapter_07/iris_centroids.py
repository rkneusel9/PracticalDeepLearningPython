import numpy as np

def centroids(x,y):
    c0 = x[np.where(y==0)].mean(axis=0)
    c1 = x[np.where(y==1)].mean(axis=0)
    c2 = x[np.where(y==2)].mean(axis=0)
    return [c0,c1,c2]

def predict(c0,c1,c2,x):
    p = np.zeros(x.shape[0], dtype="uint8")
    for i in range(x.shape[0]):
        d = [((c0-x[i])**2).sum(),
             ((c1-x[i])**2).sum(),
             ((c2-x[i])**2).sum()]
        p[i] = np.argmin(d)
    return p

def main():
    x = np.load("../data/iris/iris_features.npy")
    y = np.load("../data/iris/iris_labels.npy")
    N = 120
    x_train = x[:N]; x_test = x[N:]
    y_train = y[:N]; y_test = y[N:]
    c0, c1, c2 = centroids(x_train, y_train)
    p = predict(c0,c1,c2, x_test)
    nc = len(np.where(p == y_test)[0])
    nw = len(np.where(p != y_test)[0])
    acc = float(nc) / (float(nc)+float(nw))
    print("predicted:", p)
    print("actual   :", y_test)
    print("test accuracy = %0.4f" % acc)

main()

