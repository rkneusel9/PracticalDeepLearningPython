#
#  file:  mnist_cnn_fcn.py
#
#  Create fully convolutional version of the MNIST CNN
#  and populate with the weights from the version trained
#  on the MNIST digits.
#
#  RTK, 20-Oct-2019
#  Last update:  20-Oct-2019
#
################################################################

import keras
from keras.utils import plot_model
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

#  Load the weights from the base model
weights = load_model('mnist_cnn_base_aug_model.h5').get_weights()

#  Build the same architecture replacing Dense layers
#  with equivalent fully convolutional layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),    # input shape arbitrary
                 activation='relu',         # but grayscale
                 input_shape=(None,None,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#  Dense layer becomes Conv2D w/12x12 kernel, 128 filters
model.add(Conv2D(128, (12,12), activation='relu'))
model.add(Dropout(0.5))

#  Output layer also Conv2D but 1x1 w/10 "filters"
model.add(Conv2D(10, (1,1), activation='softmax'))

#  Copy the trained weights remapping as necessary
model.layers[0].set_weights([weights[0], weights[1]])
model.layers[1].set_weights([weights[2], weights[3]])
model.layers[4].set_weights([weights[4].reshape([12,12,64,128]), weights[5]])
model.layers[6].set_weights([weights[6].reshape([1,1,128,10]), weights[7]])

#  Output the fully convolutional model
model.save('mnist_cnn_aug_fcn_model.h5')

