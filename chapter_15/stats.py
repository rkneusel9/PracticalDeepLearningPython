import sys
import numpy as np
from keras.models import load_model

model = load_model(sys.argv[1])
x_test = np.load("../data/audio/ESC-10/esc10_spect_test_images.npy")/255.0
y_test = np.load("../data/audio/ESC-10/esc10_spect_test_labels.npy")

prob = model.predict(x_test)
p = np.argmax(prob, axis=1)

cc = np.zeros((10,10))
for i in range(len(y_test)):
    cc[y_test[i],p[i]] += 1

print()
print(np.array2string(cc.astype("uint32")))
print()

cp = 100.0 * cc / cc.sum(axis=1)
print(np.array2string(cp, precision=1))
print()

print("Overall accuracy = %0.2f%%" % (100.0*np.diag(cc).sum()/cc.sum(),))
print()

#  if second arg, output name for actual predictions
if (len(sys.argv) > 2):
    np.save(sys.argv[2], prob)

