#
#  file:  cifar10_make_animal_vehicle.py
#
#  New labels mapping CIFAR-10 images to two
#  classes - animals (1) or vehicles (0)
#
#  RTK, 20-Oct-2019
#  Last update:  20-Oct-2019
#
################################################################

import numpy as np

y_train = np.load("../data/cifar10/cifar10_train_labels.npy")
y_test  = np.load("../data/cifar10/cifar10_test_labels.npy")

for i in range(len(y_train)):
    if (y_train[i] in [0,1,8,9]):
        y_train[i] = 0
    else:
        y_train[i] = 1

for i in range(len(y_test)):
    if (y_test[i] in [0,1,8,9]):
        y_test[i] = 0
    else:
        y_test[i] = 1

np.save("../data/cifar10/cifar10_train_animal_vehicle_labels.npy", y_train)
np.save("../data/cifar10/cifar10_test_animal_vehicle_labels.npy", y_test)

