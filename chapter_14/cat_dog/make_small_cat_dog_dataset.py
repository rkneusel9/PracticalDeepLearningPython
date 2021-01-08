import numpy as np

x_train = np.load("../../data/cifar10/cifar10_train_images.npy")[:,2:30,2:30,:]
y_train = np.load("../../data/cifar10/cifar10_train_labels.npy")
x_test = np.load("../../data/cifar10/cifar10_test_images.npy")[:,2:30,2:30,:]
y_test = np.load("../../data/cifar10/cifar10_test_labels.npy")
xtrn = []; ytrn = []
xtst = []; ytst = []

for i in range(y_train.shape[0]):
    if (y_train[i]==3):
        xtrn.append(x_train[i])
        ytrn.append(0)
    if (y_train[i]==5):
        xtrn.append(x_train[i])
        ytrn.append(1)
for i in range(y_test.shape[0]):
    if (y_test[i]==3):
        xtst.append(x_test[i])
        ytst.append(0)
    if (y_test[i]==5):
        xtst.append(x_test[i])
        ytst.append(1)

np.save("../../data/cifar10/cifar10_train_cat_dog_small_images.npy", np.array(xtrn)[:1000])
np.save("../../data/cifar10/cifar10_train_cat_dog_small_labels.npy", np.array(ytrn)[:1000])
np.save("../../data/cifar10/cifar10_test_cat_dog_small_images.npy", np.array(xtst)[:1000])
np.save("../../data/cifar10/cifar10_test_cat_dog_small_labels.npy", np.array(ytst)[:1000])

