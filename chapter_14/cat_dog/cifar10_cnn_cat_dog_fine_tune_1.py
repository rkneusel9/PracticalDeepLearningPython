#
#  file:  cifar10_cnn_cat_dog_fine_tune_1.py
#
#  Fine tune the cat and dog model using the animals results
#
#  RTK, 29-Oct-2019
#  Last update:  29-Oct-2019
#
################################################################

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import numpy as np

batch_size = 64
num_classes = 2
epochs = 36
img_rows, img_cols = 28,28

#  Data prep
x_train = np.load("../../data/cifar10/cifar10_train_cat_dog_small_images.npy")
y_train = np.load("../../data/cifar10/cifar10_train_cat_dog_small_labels.npy")

x_test = np.load("../../data/cifar10/cifar10_test_cat_dog_small_images.npy")
y_test = np.load("../../data/cifar10/cifar10_test_cat_dog_small_labels.npy")

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#  Load the trained vehicles model
model = load_model("cifar10_cnn_vehicles_model.h5")

#  Change the top layer to fit the cat/dog model, 6 classes to 2 classes
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.add(Dense(num_classes, name="softmax", activation='softmax'))

#  Freeze first conv layer
model.layers[0].trainable = False
model.layers[1].trainable = True

#  Build the new model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#  As-is scores
score = model.evaluate(x_test[100:], y_test[100:], verbose=0)
print('Initial test loss:', score[0])
print('Initial test accuracy:', score[1])

#  Fine tune
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(x_test[:100], y_test[:100]))

#  Fine tuned scores
score = model.evaluate(x_test[100:], y_test[100:], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("cifar10_cnn_cat_dog_fine_tune_1_model.h5")

