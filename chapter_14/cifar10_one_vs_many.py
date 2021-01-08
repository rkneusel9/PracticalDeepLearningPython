#
#  file:  cifar10_one_vs_many.py
#
#  Compare a single multiclass model vs individual binary models
#
#  RTK, 21-Oct-2019
#  Last update:  06-Nov-2019
#
################################################################

import numpy as np
from keras.models import load_model

def main():
    x_test = np.load("../data/cifar10/cifar10_test_images.npy")/255.0
    y_test = np.load("../data/cifar10/cifar10_test_labels.npy")

    #  Load the models
    mm = load_model("cifar10_cnn_model.h5")
    m = []
    for i in range(10):
        m.append(load_model("cifar10_cnn_%d_model.h5" % i))

    #  Multiclass predictions
    mp = np.argmax(mm.predict(x_test), axis=1)

    #  Individual binary predictions
    p = np.zeros((10,10000), dtype="float32")

    for i in range(10):
        p[i,:] = m[i].predict(x_test)[:,1]

    bp = np.argmax(p, axis=0)

    #  Confusion matrices
    cm = np.zeros((10,10), dtype="uint16")
    cb = np.zeros((10,10), dtype="uint16")

    for i in range(10000):
        cm[y_test[i],mp[i]] += 1
        cb[y_test[i],bp[i]] += 1

    np.save("cifar10_multiclass_conf_mat.npy", cm)
    np.save("cifar10_binary_conf_mat.npy", cb)

    #  Confusion matrices
    print()
    print("One-vs-rest confusion matrix (rows true, cols predicted):")
    print("%s" % np.array2string(100*(cb/1000.0), precision=1))
    print()
    print("Multiclass confusion matrix:")
    print("%s"  % np.array2string(100*(cm/1000.0), precision=1))

    #  Report on differences
    db = np.diag(100*(cb/1000.0))
    dm = np.diag(100*(cm/1000.0))
    df = db - dm
    sb = np.array2string(db, precision=1)[1:-1]
    sm = np.array2string(dm, precision=1)[1:-1]
    sd = np.array2string(df, precision=1)[1:-1]
    print()
    print("Comparing per class accuracies, one-vs-rest or multiclass:")
    print()
    print("    one-vs-rest: %s" % sb)
    print("    multiclass : %s" % sm)
    print("    difference : %s" % sd)

    #  Report overall accuracies
    print()
    print("Overall accuracy:")
    print("    one-vs-rest: %0.1f%%" % np.diag(100*(cb/1000.0)).mean())
    print("    multiclass : %0.1f%%" % np.diag(100*(cm/1000.0)).mean())
    print()

main()

