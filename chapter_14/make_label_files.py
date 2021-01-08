#
#  file:  make_label_files.py
#
#  New labels mapping CIFAR-10 images to two
#  classes
#
#  RTK, 21-Oct-2019
#  Last update:  21-Oct-2019
#
################################################################

import numpy as np
import sys

if (len(sys.argv) == 1):
    print("make_label_files <class1> <train> <test>")
    exit(0)

class1 = eval("["+sys.argv[1]+"]")

y_train = np.load("../data/cifar10/cifar10_train_labels.npy")
y_test  = np.load("../data/cifar10/cifar10_test_labels.npy")

for i in range(len(y_train)):
    if (y_train[i] in class1):
        y_train[i] = 1
    else:
        y_train[i] = 0

for i in range(len(y_test)):
    if (y_test[i] in class1):
        y_test[i] = 1
    else:
        y_test[i] = 0

np.save(sys.argv[2], y_train)
np.save(sys.argv[3], y_test)

