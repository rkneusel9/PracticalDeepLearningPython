import numpy as np

def main():
    old = np.load("../data/mnist/mnist_train_labels.npy")
    new = np.zeros(len(old), dtype="uint8")
    new[np.where((old % 2) == 0)] = 0
    new[np.where((old % 2) == 1)] = 1
    np.save("../data/mnist/mnist_train_even_odd_labels.npy", new)

    old = np.load("../data/mnist/mnist_test_labels.npy")
    new = np.zeros(len(old), dtype="uint8")
    new[np.where((old % 2) == 0)] = 0
    new[np.where((old % 2) == 1)] = 1
    np.save("../data/mnist/mnist_test_even_odd_labels.npy", new)


main()

