import numpy as np
import matplotlib.pylab as plt

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def main():
    d = np.load("mnist_nn_experiments_init_results.npy")
    d0 = d[:,0,:].mean(axis=0)
    d1 = d[:,1,:].mean(axis=0)
    d2 = d[:,2,:].mean(axis=0)
    d3 = d[:,3,:].mean(axis=0)
    d4 = d[:,4,:].mean(axis=0)

    plt.plot(smooth(d0,53,"flat"), color="k", linewidth=2, linestyle="-")
    plt.plot(smooth(d1,53,"flat"), color="r", linewidth=1, linestyle="-")
    plt.plot(smooth(d2,53,"flat"), color="g", linewidth=1, linestyle="-")
    plt.plot(smooth(d3,53,"flat"), color="b", linewidth=1, linestyle="-")
    plt.plot(smooth(d4,53,"flat"), color="c", linewidth=1, linestyle="-")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Test Error", fontsize=16)
    plt.ylim((0.04,0.055))
    plt.xlim((75,4000))
    plt.tight_layout()
    plt.savefig("mnist_nn_experiments_init_plot.png", type="png", dpi=600)
    plt.savefig("mnist_nn_experiments_init_plot.pdf", type="pdf", dpi=600)
    plt.show()


main()

