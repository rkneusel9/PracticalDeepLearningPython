#
#  file:  mnist_cnn_fcn_test.py
#
#  Test the fully convolutional version
#
#  RTK, 20-Oct-2019
#  Last update:  20-Oct-2019
#
################################################################

import os
import time
import numpy as np
from keras.models import load_model
from PIL import Image

def main():
    #  Test individual MNIST digits
    x_test = np.load("../../../../data/mnist/mnist_test_images.npy")/255.0
    y_test = np.load("../../../../data/mnist/mnist_test_labels.npy")

    #  Random subset
    N = 1000
    idx = np.argsort(np.random.random(N))

    #  The FCN model
    model = load_model("mnist_cnn_aug_fcn_model.h5")

    #  Make predictions on the digits and track accuracy
    nc = nw = 0.0

    for i in idx:
        p = model.predict(x_test[i][np.newaxis,:,:,np.newaxis])
        c = np.argmax(p)
        if (c == y_test[i]):
            nc += 1
        else:
            nw += 1
    print()
    print("Single MNIST digits, n=%d, accuracy = %0.2f%%" % (N, 100*nc/N))
    print()

    #  Test larger digit images and store the results
    os.system("rm -rf results_aug; mkdir results_aug")
    n = len(os.listdir("images"))

    print("Processing larger digit images... ")
    st = time.time()
    for i in range(n):
        f = "images/image_%04d.png" % i
        im = np.array(Image.open(f))/255.0
        p = model.predict(im[np.newaxis,:,:,np.newaxis])
        np.save("results_aug/results_%04d.npy" % i, p[0,:,:,:])
    print("done, time = %0.3f s" % (time.time()-st))
    print()


main()

