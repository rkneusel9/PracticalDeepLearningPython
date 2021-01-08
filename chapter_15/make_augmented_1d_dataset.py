#
#  file:  make_augmented_1d_dataset.py
#
#  RTK, 13-Nov-2019
#  Last update:  13-Nov-2019
#
#  Use the augmented .wav files.
#
################################################################

import os
import random
import numpy as np
from scipy.io.wavfile import read

sr = 44100 # Hz
N = 2*sr   # number of samples to keep
w = 100    # every 100 (0.01 s)

#  train
afiles = [i[:-1] for i in open("../data/audio/ESC-10/augmented_train_filelist.txt")]
trn = np.zeros((len(afiles),N//w,1), dtype="int16") 
lbl = np.zeros(len(afiles), dtype="uint8")

for i,t in enumerate(afiles):
    f,c = t.split()
    trn[i,:,0] = read(f)[1][:N:w]
    lbl[i] = int(c)

np.save("../data/audio/ESC-10/esc10_raw_train_audio.npy", trn)
np.save("../data/audio/ESC-10/esc10_raw_train_labels.npy", lbl)

#  test
afiles = [i[:-1] for i in open("../data/audio/ESC-10/augmented_test_filelist.txt")]
tst = np.zeros((len(afiles),N//w,1), dtype="int16") 
lbl = np.zeros(len(afiles), dtype="uint8")

for i,t in enumerate(afiles):
    f,c = t.split()
    tst[i,:,0] = read(f)[1][:N:w]
    lbl[i] = int(c)

np.save("../data/audio/ESC-10/esc10_raw_test_audio.npy", tst)
np.save("../data/audio/ESC-10/esc10_raw_test_labels.npy", lbl)

