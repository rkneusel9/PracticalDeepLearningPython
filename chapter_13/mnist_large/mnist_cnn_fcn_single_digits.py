import numpy as np
from keras.models import load_model

x_test = np.load("../../../../data/mnist/mnist_test_images.npy")/255.0
y_test = np.load("../../../../data/mnist/mnist_test_labels.npy")

model = load_model("mnist_cnn_fcn_model.h5")
N = y_test.shape[0]
nc = nw = 0.0 
for i in range(N):
    p = model.predict(x_test[i][np.newaxis,:,:,np.newaxis])
    c = np.argmax(p)
    if (c == y_test[i]):
        nc += 1
    else:
        nw += 1
print()
print("Single MNIST digits, n=%d, accuracy = %0.2f%%" % (N, 100*nc/N))
print()

