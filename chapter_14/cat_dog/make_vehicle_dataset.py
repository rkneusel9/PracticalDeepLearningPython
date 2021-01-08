import numpy as np

x_train = np.load("../../data/cifar10/cifar10_aug_train_images.npy")
y_train = np.load("../../data/cifar10/cifar10_aug_train_labels.npy")
x_test = np.load("../../data/cifar10/cifar10_aug_test_images.npy")
y_test = np.load("../../data/cifar10/cifar10_test_labels.npy")

vehicles= [0,1,8,9]
xv_train = []; xv_test = []
yv_train = []; yv_test = []

for i in range(y_train.shape[0]):
    if (y_train[i] in vehicles):
        xv_train.append(x_train[i])
        yv_train.append(vehicles.index(y_train[i]))
for i in range(y_test.shape[0]):
    if (y_test[i] in vehicles):
        xv_test.append(x_test[i])
        yv_test.append(vehicles.index(y_test[i]))

np.save("../../data/cifar10/cifar10_train_vehicles_images.npy", np.array(xv_train))
np.save("../../data/cifar10/cifar10_train_vehicles_labels.npy", np.array(yv_train))
np.save("../../data/cifar10/cifar10_test_vehicles_images.npy", np.array(xv_test))
np.save("../../data/cifar10/cifar10_test_vehicles_labels.npy", np.array(yv_test))

