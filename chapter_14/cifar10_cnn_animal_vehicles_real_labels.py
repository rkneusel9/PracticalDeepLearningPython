#
#  file:  cifar10_cnn_animal_vehicles_real_labels.py
#
#  Report on animal vs vehicles predictions and actual
#  label of errors.
#
#  RTK, 05-Nov-2019
#  Last update:  05-Nov-2019
#
################################################################

import numpy as np
from keras.models import load_model

x_test = np.load("../data/cifar10/cifar10_test_images.npy")/255.0
y_label= np.load("../data/cifar10/cifar10_test_labels.npy")
y_test = np.load("../data/cifar10_test_animal_vehicles_labels.npy")
model = load_model("../data/cifar10_cnn_animal_vehicles_model.h5")

pp = model.predict(x_test)
p = np.zeros(pp.shape[0], dtype="uint8")
for i in range(pp.shape[0]):
    p[i] = 0 if (pp[i,0] > pp[i,1]) else 1

hp = []
hn = []
for i in range(len(y_test)):
    if (p[i] == 0) and (y_test[i] == 1): 
        hn.append(y_label[i])   
    elif (p[i] == 1) and (y_test[i] == 0): 
        hp.append(y_label[i])
hp = np.array(hp)
hn = np.array(hn)
a = np.histogram(hp, bins=10, range=[0,9])[0]
b = np.histogram(hn, bins=10, range=[0,9])[0]
print("vehicles as animals: %s" % np.array2string(a))
print("animals as vehicles: %s" % np.array2string(b))

