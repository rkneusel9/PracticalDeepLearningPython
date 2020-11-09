import os
import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model("mnist_cnn_fcn_model.h5")

os.system("rm -rf results; mkdir results")
n = len(os.listdir("images"))

for i in range(n):
    f = "images/image_%04d.png" % i
    im = np.array(Image.open(f))/255.0
    p = model.predict(im[np.newaxis,:,:,np.newaxis])
    np.save("results/results_%04d.npy" % i, p[0,:,:,:])

