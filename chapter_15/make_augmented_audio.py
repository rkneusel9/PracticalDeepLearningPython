#
#  file:  make_augmented_audio.py
#
#  Augment the raw audio files
#
#  RTK, 10-Nov-2019
#  Last update:  11-Nov-2019
#
################################################################

import os
import random
import numpy as np
from scipy.io.wavfile import read, write
import librosa as rosa

#  apply augmentations
def augment(wav):
    sr = wav[0]
    d = wav[1].astype("float32")
    if (random.random() < 0.5):
        s = int(sr/4.0*(np.random.random()-0.5))
        d = np.roll(d,s)
        if (s < 0):
            d[s:] = 0
        else:
            d[:s] = 0
    if (random.random() < 0.5):
        #  noise
        d += 0.1*(d.max()-d.min())*np.random.random(d.shape[0])
    if (random.random() < 0.5):
        #  pitch shift
        pf = 20.0*(np.random.random()-0.5)
        d = rosa.effects.pitch_shift(d, sr, pf)
    if (random.random() < 0.5):
        #  time stretch
        rate = 1.0 + (np.random.random()-0.5)
        d = rosa.effects.time_stretch(d,rate)
        if (d.shape[0] > wav[1].shape[0]):
            d = d[:wav[1].shape[0]]
        else:
            w = np.zeros(wav[1].shape[0], dtype="float32")
            w[:d.shape[0]] = d
            d = w.copy()
    return d

def augment_audio(src_list, typ):
    flist = []
    for i,s in enumerate(src_list):
        f,c = s.split()
        wav = read(f) # (sample rate, data, type)
        base = os.path.abspath("../data/audio/ESC-10/augmented/%s/%s" % (typ, os.path.basename(f)[:-4]))
        fname = base+".wav"
        write(fname, wav[0], wav[1])
        flist.append("%s %s" % (fname,c))
        for j in range(19):
            d = augment(wav)
            fname = base+("_%04d.wav" % j)
            write(fname, wav[0], d.astype(wav[1].dtype))
            flist.append("%s %s" % (fname,c))

    random.shuffle(flist)
    with open("../data/audio/ESC-10/augmented_%s_filelist.txt" % typ,"w") as f:
        for z in flist:
            f.write("%s\n" % z)

def main():
    #  Number of test cases / class
    N = 8  # 20%

    #  output directory (overwrite)
    os.system("rm -rf ../data/audio/ESC-10/augmented; mkdir ../data/audio/ESC-10/augmented")
    os.system("mkdir ../data/audio/ESC-10/augmented/train ../data/audio/ESC-10/augmented/test")

    #  original .wav files
    src_list = [i[:-1] for i in open("../data/audio/ESC-10/filelist.txt")]

    #  split into train/test
    z = [[] for i in range(10)]
    for s in src_list:
        _,c = s.split()
        z[int(c)].append(s)

    train = []
    test = []
    for i in range(10):
        p = z[i]
        random.shuffle(p)
        test += p[:8]
        train += p[8:]

    random.shuffle(train)
    random.shuffle(test)

    #  augment
    augment_audio(train, "train")
    augment_audio(test, "test")

main()

