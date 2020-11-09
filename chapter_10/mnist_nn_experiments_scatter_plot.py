#
#  file:  mnist_nn_experiments_scatter_plot.py
#
#  ReLU results scatter plot with marker size the score.
#
#  RTK, 30-Dec-2018
#  Last update:  16-Dec-2019
#
###############################################################

import matplotlib.pylab as plt
import numpy as np

def main():
    one = np.array([0.8656,0.8683,0.8716,0.8698])
    e1 = np.array([0.0007,0.0006,0.0006,0.0006])

    two = np.array([0.8748,0.8781,0.8808,0.8798])
    e2 = np.array([0.0010,0.0007,0.0006,0.0006])

    three=np.array([0.8778,0.8822,0.8848,0.8864])
    e3 = np.array([0.0011,0.0008,0.0006,0.0007])

    p1 = np.array([795010,1590010,3180010,6360010])
    p2 = np.array([798360,1570335,3173685,6314185])
    p3 = np.array([792505,1580320,3187627,6355475])

    plt.errorbar(p1,one,e1,marker="o",color='red',label='One')
    plt.errorbar(p2,two,e2,marker='s',color='blue',label='Two')
    plt.errorbar(p3,three,e3,marker='^',color='green',label='Three')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.xlabel("Parameters")
    plt.ylabel("Score")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("relu_layers.pdf", format="pdf", dpi=600)
    plt.show()


main()

#  use these: (25 models averaged)
#layers:        (1000,), score= 0.8656 +/- 0.0007, loss = 0.3306 +/- 0.0006 (params = 795010, time = 27.88 s)
#layers:        (2000,), score= 0.8683 +/- 0.0006, loss = 0.3193 +/- 0.0004 (params = 1590010, time = 58.31 s)
#layers:        (4000,), score= 0.8716 +/- 0.0006, loss = 0.3120 +/- 0.0002 (params = 3180010, time = 112.95 s)
#layers:        (8000,), score= 0.8698 +/- 0.0006, loss = 0.3075 +/- 0.0001 (params = 6360010, time = 272.07 s)
#layers:     (700, 350), score= 0.8748 +/- 0.0010, loss = 0.2392 +/- 0.0009 (params = 798360, time = 28.90 s)
#layers:    (1150, 575), score= 0.8781 +/- 0.0007, loss = 0.2278 +/- 0.0006 (params = 1570335, time = 56.44 s)
#layers:    (1850, 925), score= 0.8808 +/- 0.0006, loss = 0.2179 +/- 0.0006 (params = 3173685, time = 115.29 s)
#layers:   (2850, 1425), score= 0.8798 +/- 0.0006, loss = 0.2116 +/- 0.0005 (params = 6314185, time = 231.93 s)
#layers: (660, 330, 165), score= 0.8778 +/- 0.0011, loss = 0.1810 +/- 0.0016 (params = 792505, time = 29.01 s)
#layers: (1080, 540, 270), score= 0.8822 +/- 0.0008, loss = 0.1637 +/- 0.0010 (params = 1580320, time = 57.30 s)
#layers: (1714, 857, 429), score= 0.8848 +/- 0.0006, loss = 0.1534 +/- 0.0007 (params = 3187627, time = 115.90 s)
#layers: (2620, 1310, 655), score= 0.8864 +/- 0.0007, loss = 0.1460 +/- 0.0006 (params = 6355475, time = 229.24 s)

