import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

p0 = np.load("prob_run0.npy")
p1 = np.load("prob_run1.npy")
p2 = np.load("prob_run2.npy")
p3 = np.load("prob_run3.npy")
p4 = np.load("prob_run4.npy")
p5 = np.load("prob_run5.npy")

y_test = np.load("../data/audio/ESC-10/esc10_spect_test_labels.npy")

prob = (p0+p1+p2+p3+p4+p5)/6.0
p = np.argmax(prob, axis=1)

cc = np.zeros((10,10))
for i in range(len(y_test)):
    cc[y_test[i],p[i]] += 1

print()
print("Ensemble average:")
print()

if (len(sys.argv) > 1):
    print(np.array2string(cc.astype("uint32")))
    print()

    cp = 100.0 * cc / cc.sum(axis=1)
    print(np.array2string(cp, precision=1))
    print()

print("Overall accuracy = %0.2f%%" % (100.0*np.diag(cc).sum()/cc.sum(),))
print()

print()
print("Ensemble max:")
print()

p = np.zeros(len(y_test), dtype="uint8")
for i in range(len(y_test)):
    mx = 0.0
    idx = 0
    t = np.array([p0[i],p1[i],p2[i],p3[i],p4[i],p5[i]])
    p[i] = np.argmax(t.reshape(60)) % 10

cc = np.zeros((10,10))
for i in range(len(y_test)):
    cc[y_test[i],p[i]] += 1

if (len(sys.argv) > 1):
    print(np.array2string(cc.astype("uint32")))
    print()

    cp = 100.0 * cc / cc.sum(axis=1)
    print(np.array2string(cp, precision=1))
    print()

print("Overall accuracy = %0.2f%%" % (100.0*np.diag(cc).sum()/cc.sum(),))
print()

#  Voting
print()
print("Ensemble voting:")
print()
t = np.zeros((6,len(y_test)), dtype="uint32")
t[0,:] = np.argmax(p0, axis=1)
t[1,:] = np.argmax(p1, axis=1)
t[2,:] = np.argmax(p2, axis=1)
t[3,:] = np.argmax(p3, axis=1)
t[4,:] = np.argmax(p4, axis=1)
t[5,:] = np.argmax(p5, axis=1)
p = np.zeros(len(y_test), dtype="uint8")
for i in range(len(y_test)):
    q = np.bincount(t[:,i])
    p[i] = np.argmax(q)

cc = np.zeros((10,10))
for i in range(len(y_test)):
    cc[y_test[i],p[i]] += 1

if (len(sys.argv) > 1):
    print(np.array2string(cc.astype("uint32")))
    print()

    cp = 100.0 * cc / cc.sum(axis=1)
    print(np.array2string(cp, precision=1))
    print()

print("Overall accuracy = %0.2f%%" % (100.0*np.diag(cc).sum()/cc.sum(),))
print()

