import matplotlib.pylab as plt

def main():
    x = [100, 200, 300, 400, 500, 600,700, 800, 900, 1000, 
        1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 7500, 
        10000, 15000, 20000, 25000, 30000]
    y = [0.75330, 0.81608, 0.85652, 0.86402, 0.87372, 0.87764,
        0.88382, 0.89334, 0.89644, 0.90168, 0.91856, 0.92490,
        0.92966, 0.93690, 0.94190, 0.94424, 0.94964, 0.95032,
        0.95862, 0.96526, 0.96664, 0.96944, 0.96980, 0.96926]
  
    plt.plot(x,y, marker="o", color="b")
    plt.xlabel("Number of training samples")
    plt.ylabel("Mean test score")
    plt.tight_layout()
    plt.savefig("mnist_nn_experiments_samples_plot.pdf", type="pdf", dpi=600)
    plt.show()


main()

