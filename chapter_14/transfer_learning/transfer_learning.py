#
#  file:  transfer_learning.py
#
#  RTK, 03-Nov-2019
#  Last update:  27-Mar-2022
#
################################################################

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

def conf_mat(clf,x,y):
    p = clf.predict(x)
    c = np.zeros((10,10))
    for i in range(p.shape[0]):
        c[y[i],p[i]] += 1
    return c

#  load the dataset - MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

#  and the model
model = load_model("cifar10_cnn_model.h5")

#  run each training sample through the CIFAR-10 model
print()
print("Running the MNIST training images through the model")
train = np.zeros((60000,128))
k = 0
for i in range(600):
    t = np.zeros((100,32,32,3))
    t[:,2:30,2:30,0] = x_train[k:(k+100)]
    t[:,2:30,2:30,1] = x_train[k:(k+100)]
    t[:,2:30,2:30,2] = x_train[k:(k+100)]
    _ = model.predict(t)
    out = [model.layers[5].output]
    func = K.function([model.input], out)
    train[k:(k+100),:] = func([t])[0]
    k += 100
np.save("mnist_train_embedded.npy", train)

#  do the same for the test set
print("Running the MNIST test images through the model")
test = np.zeros((10000,128))
k = 0
for i in range(100):
    t = np.zeros((100,32,32,3))
    t[:,2:30,2:30,0] = x_test[k:(k+100)]
    t[:,2:30,2:30,1] = x_test[k:(k+100)]
    t[:,2:30,2:30,2] = x_test[k:(k+100)]
    _ = model.predict(t)
    out = [model.layers[5].output]
    func = K.function([model.input], out)
    test[k:(k+100),:] = func([t])[0]
    k += 100
np.save("mnist_test_embedded.npy", test)

#  use the 128 element vectors as training data for other models
print("Full MNIST dataset:")
print()
print("Training Nearest Centroid")
clf0 = NearestCentroid()
clf0.fit(train, y_train)
nscore = 100.0*clf0.score(test, y_test)

print("Training 3-NN")
clf1 = KNeighborsClassifier(n_neighbors=3)
clf1.fit(train, y_train)
kscore = 100.0*clf1.score(test, y_test)

print("Training Random Forest")
clf2 = RandomForestClassifier(n_estimators=50)
clf2.fit(train, y_train)
rscore = 100.0*clf2.score(test, y_test)

print("Training Linear SVM")
clf3 = LinearSVC(C=0.1)
clf3.fit(train, y_train)
sscore = 100.0*clf3.score(test, y_test)

#  report transfer learning results
print()
print("Nearest Centroid    : %0.2f" % nscore)
print("3-NN                : %0.2f" % kscore)
print("Random Forest       : %0.2f" % rscore)
print("SVM                 : %0.2f" % sscore)
print()

#  confusion matrices
cn = conf_mat(clf0, test, y_test) 
ck = conf_mat(clf1, test, y_test)
cr = conf_mat(clf2, test, y_test)
cs = conf_mat(clf3, test, y_test)
cn = 100.0*cn / cn.sum(axis=1)
ck = 100.0*ck / ck.sum(axis=1)
cr = 100.0*cr / cr.sum(axis=1)
cs = 100.0*cs / cs.sum(axis=1)
np.save("confusion_nearest.npy", cn)
np.save("confusion_3NN.npy", ck)
np.save("confusion_random.npy", cr)
np.save("confusion_svm.npy", cs)

np.set_printoptions(suppress=True)
print()
print("Nearest Centroid:")
print(np.array2string(cn, precision=1, floatmode="fixed"))
print()
print("3-NN:")
print(np.array2string(ck, precision=1, floatmode="fixed"))
print()
print("Random Forest:")
print(np.array2string(cr, precision=1, floatmode="fixed"))
print()
print("SVM:")
print(np.array2string(cs, precision=1, floatmode="fixed"))
print()

#  train directly from the images
x_train = x_train.reshape((60000,28*28))
x_test = x_test.reshape((10000,28*28))

clf0 = NearestCentroid()
clf0.fit(x_train, y_train)
nscore = 100.0*clf0.score(x_test, y_test)

print("Training 3-NN")
clf1 = KNeighborsClassifier(n_neighbors=3)
clf1.fit(x_train, y_train)
kscore = 100.0*clf1.score(x_test, y_test)

print("Training Random Forest")
clf2 = RandomForestClassifier(n_estimators=50)
clf2.fit(x_train, y_train)
rscore = 100.0*clf2.score(x_test, y_test)

print("Training Linear SVM")
clf3 = LinearSVC(C=0.1)
clf3.fit(x_train, y_train)
sscore = 100.0*clf3.score(x_test, y_test)

#  report transfer learning results
print()
print("Image Nearest Centroid    : %0.2f" % nscore)
print("Image 3-NN                : %0.2f" % kscore)
print("Image Random Forest       : %0.2f" % rscore)
print("Image SVM                 : %0.2f" % sscore)
print()

#  confusion matrices
cn = conf_mat(clf0, x_test, y_test) 
ck = conf_mat(clf1, x_test, y_test)
cr = conf_mat(clf2, x_test, y_test)
cs = conf_mat(clf3, x_test, y_test)
cn = 100.0*cn / cn.sum(axis=1)
ck = 100.0*ck / ck.sum(axis=1)
cr = 100.0*cr / cr.sum(axis=1)
cs = 100.0*cs / cs.sum(axis=1)
np.save("confusion_image_nearest.npy", cn)
np.save("confusion_image_3NN.npy", ck)
np.save("confusion_image_random.npy", cr)
np.save("confusion_image_svm.npy", cs)

np.set_printoptions(suppress=True)
print()
print("Image Nearest Centroid:")
print(np.array2string(cn, precision=1, floatmode="fixed"))
print()
print("Image 3-NN:")
print(np.array2string(ck, precision=1, floatmode="fixed"))
print()
print("Image Random Forest:")
print(np.array2string(cr, precision=1, floatmode="fixed"))
print()
print("Image SVM:")
print(np.array2string(cs, precision=1, floatmode="fixed"))
print()

