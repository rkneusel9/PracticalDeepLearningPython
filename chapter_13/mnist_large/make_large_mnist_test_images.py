#
#  file: make_large_mnist_test_images.py
#
#  Make some large MNIST test images.
#
#  RTK, 19-Oct-2019
#  Last update:  20-Oct-2019
#
################################################################

import os
import sys
import numpy as np
import random
from PIL import Image

def main():
    """Make large test images"""

    os.system("rm -rf images; mkdir images")

    if (len(sys.argv) > 1):
        N = int(sys.argv[1])
    else:
        N = 10

    x_test = np.load("../../data/mnist/mnist_test_images.npy")

    for i in range(N):
        r,c = random.randint(6,12), random.randint(6,12)
        g = np.zeros(r*c)
        for j in range(r*c):
            if (random.random() < 0.15):
                g[j] = 1
        g = g.reshape((r,c))
        g[:,0] = g[0,:] = g[:,-1] = g[-1,:] = 0

        img = np.zeros((28*r,28*c), dtype="uint8")
        for x in range(r):
            for y in range(c):
                if (g[x,y] == 1):
                    n = random.randint(0, x_test.shape[0])
                    im = x_test[n]
                    img[28*x:(28*x+28), 28*y:(28*y+28)] = im
        
        Image.fromarray(img).save("images/image_%04d.png" % i)


if (__name__ == "__main__"):
    main()

