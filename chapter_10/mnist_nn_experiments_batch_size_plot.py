import numpy as np
import matplotlib.pylab as plt

def main():
    # epochs == 100
    bz = np.array([2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384])
    sc = np.array([0.97174,0.96976,0.96876,0.96734,0.96596,0.96332,0.95684,0.94564,0.93312,0.91788,0.90046,0.87318,0.82926,0.76104])
    ec = np.array([0.00006,0.00034,0.00007,0.00012,0.00011,0.00040,0.00042,0.00040,0.00038,0.00040,0.00078,0.00086,0.00203,0.00591])

    # minibatches = 8192
    sc0= np.array([0.93658,0.94556,0.94856,0.94916,0.95012,0.94946,0.95068,0.95038,0.95112,0.95030,0.95066,0.95028,0.94992,0.94994])
    ec0= np.array([0.00214,0.00078,0.00070,0.00115,0.00025,0.00028,0.00053,0.00041,0.00045,0.00023,0.00032,0.00058,0.00044,0.00022])

    plt.errorbar(bz,sc,ec,marker='o',color='red')
    plt.errorbar(bz,sc0,ec0,marker='s',color='blue')
    plt.xlabel("Minibatch Size")
    plt.ylabel("Mean Score")
    plt.tight_layout()
    plt.savefig("mnist_nn_experiments_batch_size_plot.pdf", format="pdf", dpi=600)
    plt.show()


main()

