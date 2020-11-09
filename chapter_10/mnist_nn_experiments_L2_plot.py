import numpy as np
import matplotlib.pylab as plt

def main():
    d0 = np.load("mnist_nn_experiments_L2/val_error_0.000000.npy")
    d1 = np.load("mnist_nn_experiments_L2/val_error_0.100000.npy")
    d2 = np.load("mnist_nn_experiments_L2/val_error_0.200000.npy")
    d3 = np.load("mnist_nn_experiments_L2/val_error_0.300000.npy")
    d4 = np.load("mnist_nn_experiments_L2/val_error_0.400000.npy")

    plt.plot(d0, color="k", linewidth=1, linestyle="-")
    plt.plot(d1, color="r", linewidth=1, linestyle="-")
    plt.plot(d2, color="g", linewidth=1, linestyle="-")
    plt.plot(d3, color="b", linewidth=1, linestyle="-")
    plt.plot(d4, color="c", linewidth=1, linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Error")
    plt.ylim((0.05,0.1))
    plt.tight_layout()
    plt.savefig("mnist_nn_experiments_L2_plot.pdf", type="pdf", dpi=600)
    plt.show()


main()

