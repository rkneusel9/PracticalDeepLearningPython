#
#  file:  make_augmented_spectrograms.py
#
#  Use sox to make the spectrogram images.
#
#  RTK, 11-Nov-2019
#  Last update: 11-Nov-2019
#
################################################################

import os
import numpy as np
from PIL import Image

rows = 100
cols = 160

# train
flist = [i[:-1] for i in open("../data/audio/ESC-10/augmented_train_filelist.txt")]
N = len(flist)
img = np.zeros((N,rows,cols,3), dtype="uint8")
lbl = np.zeros(N, dtype="uint8")
p = []

for i,f in enumerate(flist):
    src, c = f.split()
    os.system("sox %s -n spectrogram" % src)
    im = np.array(Image.open("spectrogram.png").convert("RGB"))
    im = im[42:542,58:858,:]
    im = Image.fromarray(im).resize((cols,rows))
    img[i,:,:,:] = np.array(im)
    lbl[i] = int(c)
    p.append(os.path.abspath(src))

os.system("rm -rf spectrogram.png")
p = np.array(p)
idx = np.argsort(np.random.random(N))
img = img[idx]
lbl = lbl[idx]
p = p[idx]
np.save("../data/audio/ESC-10/esc10_spect_train_images.npy", img)
np.save("../data/audio/ESC-10/esc10_spect_train_labels.npy", lbl)
np.save("../data/audio/ESC-10/esc10_spect_train_paths.npy", p)

# test
flist = [i[:-1] for i in open("../data/audio/ESC-10/augmented_test_filelist.txt")]
N = len(flist)
img = np.zeros((N,rows,cols,3), dtype="uint8")
lbl = np.zeros(N, dtype="uint8")
p = []

for i,f in enumerate(flist):
    src, c = f.split()
    os.system("sox %s -n spectrogram" % src)
    im = np.array(Image.open("spectrogram.png").convert("RGB"))
    im = im[42:542,58:858,:]
    im = Image.fromarray(im).resize((cols,rows))
    img[i,:,:,:] = np.array(im)
    lbl[i] = int(c)
    p.append(os.path.abspath(src))

os.system("rm -rf spectrogram.png")
p = np.array(p)
idx = np.argsort(np.random.random(N))
img = img[idx]
lbl = lbl[idx]
p = p[idx]
np.save("../data/audio/ESC-10/esc10_spect_test_images.npy", img)
np.save("../data/audio/ESC-10/esc10_spect_test_labels.npy", lbl)
np.save("../data/audio/ESC-10/esc10_spect_test_paths.npy", p)

