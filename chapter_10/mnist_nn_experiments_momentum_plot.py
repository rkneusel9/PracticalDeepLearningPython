import numpy as np
import matplotlib.pylab as plt

def main():
    d0 = np.load("mnist_nn_experiments_momentum/val_error_0.00.npy")
    d1 = np.load("mnist_nn_experiments_momentum/val_error_0.30.npy")
    d2 = np.load("mnist_nn_experiments_momentum/val_error_0.50.npy")
    d3 = np.load("mnist_nn_experiments_momentum/val_error_0.70.npy")
    d4 = np.load("mnist_nn_experiments_momentum/val_error_0.90.npy")
    d5 = np.load("mnist_nn_experiments_momentum/val_error_0.99.npy")

    plt.plot(d0, color="k", linewidth=1, linestyle="-")
    plt.plot(d1, color="r", linewidth=1, linestyle="-")
    plt.plot(d2, color="g", linewidth=1, linestyle="-")
    plt.plot(d3, color="b", linewidth=1, linestyle="-")
    plt.plot(d4, color="c", linewidth=1, linestyle="-")
    plt.plot(d5, color="m", linewidth=1, linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Test Error")
    plt.ylim((0.05,0.1))
    plt.tight_layout()
    plt.savefig("mnist_nn_experiments_momentum_plot.pdf", type="pdf", dpi=600)
    plt.savefig("mnist_nn_experiments_momentum_plot.png", type="png", dpi=600)
    plt.show()


main()

