#
#  file:  make_shifted_mnist_dataset.py
#
#  Augment by shifting for use with FCN
#
#  RTK, 20-Oct-2019
#  Last update:  20-Oct-2019
#
################################################################

import numpy as np
import random

def shifted(im):
    r,c = im.shape
    x = random.randint(-r//4, r//4)
    y = random.randint(-c//4, c//4)
    img = np.zeros((2*r,2*c), dtype="uint8")
    xoff = r//2 + x
    yoff = c//2 + y
    img[xoff:(xoff+r), yoff:(yoff+c)] = im
    img = img[r//2:(r//2+r),c//2:(c//2+c)]
    return img


def main():
    x_train = np.load("../../../../data/mnist/mnist_train_images.npy")
    x_test = np.load("../../../../data/mnist/mnist_test_images.npy")

    y_train = np.load("../../../../data/mnist/mnist_train_labels.npy") 
    y_test = np.load("../../../../data/mnist/mnist_test_labels.npy")

    x_train_aug = np.zeros((5*60000,28,28), dtype="uint8")
    x_test_aug = np.zeros((5*10000,28,28), dtype="uint8")
    y_train_labels = np.zeros(5*60000, dtype="uint8")
    y_test_labels = np.zeros(5*10000, dtype="uint8")

    k = 0
    for i in range(60000):
        x_train_aug[k] = x_train[i]
        y_train_labels[k] = y_train[i]
        k += 1
        for j in range(4):
            x_train_aug[k] = shifted(x_train[i])
            y_train_labels[k] = y_train[i]
            k += 1

    k = 0
    for i in range(10000):
        x_test_aug[k] = x_test[i]
        y_test_labels[k] = y_test[i]
        k += 1
        for j in range(4):
            x_test_aug[k] = shifted(x_test[i])
            y_test_labels[k] = y_test[i]
            k += 1

    idx = np.argsort(np.random.random(5*60000))
    x_train_aug = x_train_aug[idx]
    y_train_labels = y_train_labels[idx]

    idx = np.argsort(np.random.random(5*10000))
    x_test_aug = x_test_aug[idx]
    y_test_labels = y_test_labels[idx]

    np.save("mnist_train_aug_images.npy", x_train_aug)
    np.save("mnist_train_aug_labels.npy", y_train_labels)
    np.save("mnist_test_aug_images.npy", x_test_aug)
    np.save("mnist_test_aug_labels.npy", y_test_labels)


main()

