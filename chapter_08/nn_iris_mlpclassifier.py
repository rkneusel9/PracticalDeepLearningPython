import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier

xtrain= np.load("../data/iris/iris2_train.npy")
ytrain= np.load("../data/iris/iris2_train_labels.npy")
xtest = np.load("../data/iris/iris2_test.npy")
ytest = np.load("../data/iris/iris2_test_labels.npy")

clf = MLPClassifier(
        hidden_layer_sizes=(3,2),
        activation="logistic",
        solver="adam", tol=1e-9,
        max_iter=5000,
        verbose=True)
clf.fit(xtrain, ytrain)
prob = clf.predict_proba(xtest)
score = clf.score(xtest, ytest)

w12 = clf.coefs_[0]
w23 = clf.coefs_[1]
w34 = clf.coefs_[2]
b1 = clf.intercepts_[0]
b2 = clf.intercepts_[1]
b3 = clf.intercepts_[2]
weights = [w12,b1,w23,b2,w34,b3]
pickle.dump(weights, open("../data/iris/iris2_weights.pkl","wb"))

print()
print("Test results:")
print("  Overall score: %0.7f" % score)
print()
for i in range(len(ytest)):
    p = 0 if (prob[i,1] < 0.5) else 1
    print("%03d: %d - %d, %0.7f" % (i, ytest[i], p, prob[i,1]))
print()

