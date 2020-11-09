import numpy as np
import matplotlib.pylab as plt

def main():
    # score by base_lr, fixed epochs
    sc0 = np.array([0.91870,0.95070,0.96050,0.97120,0.97260,0.97540,0.97630,0.94800])
    lr = np.array([0.00010,0.00050,0.00100,0.00500,0.01000,0.05000,0.10000,0.20000])

    # score by base_lr, lr * epochs = 1.5
    sc1 = np.array([0.96990,0.97030,0.97060,0.97240,0.97310,0.97590,0.97340,0.95550])

    plt.semilogx(lr,sc0,marker='o',color='red')
    plt.semilogx(lr,sc1,marker='s',color='blue')
    plt.xlabel("Learning rate ($\eta$)")
    plt.ylabel("Test Score")
    plt.tight_layout()
    plt.savefig("mnist_nn_experiments_base_lr_plot.pdf", format="pdf", dpi=600)
    plt.show()

main()


#base_lr = 0.20000, score = 0.94800, loss = 0.09340, epochs = 50
#base_lr = 0.10000, score = 0.97630, loss = 0.00207, epochs = 50
#base_lr = 0.05000, score = 0.97540, loss = 0.00162, epochs = 50
#base_lr = 0.01000, score = 0.97260, loss = 0.00229, epochs = 50
#base_lr = 0.00500, score = 0.97120, loss = 0.00413, epochs = 50
#base_lr = 0.00100, score = 0.96050, loss = 0.06542, epochs = 50
#base_lr = 0.00050, score = 0.95070, loss = 0.13361, epochs = 50
#base_lr = 0.00010, score = 0.91870, loss = 0.29111, epochs = 50
#
#base_lr = 0.20000, score = 0.95550, loss = 0.07414, epochs = 8, time = 71.445
#base_lr = 0.10000, score = 0.97340, loss = 0.00946, epochs = 15, time = 132.122
#base_lr = 0.05000, score = 0.97590, loss = 0.00168, epochs = 30, time = 268.132
#base_lr = 0.01000, score = 0.97310, loss = 0.00163, epochs = 150, time = 1385.001
#base_lr = 0.00500, score = 0.97240, loss = 0.00163, epochs = 300, time = 2741.335
#base_lr = 0.00100, score = 0.97060, loss = 0.00163, epochs = 1500, time = 13232.182
#base_lr = 0.00050, score = 0.97030, loss = 0.00163, epochs = 3000, time = 26478.357
#base_lr = 0.00010, score = 0.96990, loss = 0.00162, epochs = 15000, time = 135642.231


