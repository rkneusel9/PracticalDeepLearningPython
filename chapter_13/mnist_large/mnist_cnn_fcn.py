#
#  file:  mnist_cnn_fcn.py
#
#  Create fully convolutional version of the MNIST CNN
#  and populate with the weights from the version trained
#  on the MNIST digits.
#
#  RTK, 20-Oct-2019
#  Last update:  04-Mar-2022
#
################################################################

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#  Load the weights from the base model
weights = load_model('mnist_cnn_base_model.h5').get_weights()

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
model.compile(optimizer='adam', loss='binary_crossentropy')
model.save('mnist_cnn_fcn_model.h5')

