import sys
import os
import shutil

def main():
    """Extract the 10 class subset"""

    classes = {
        "rain":0,
        "rooster":1,
        "crying_baby":2,
        "sea_waves":3,
        "clock_tick":4,
        "sneezing":5,
        "dog":6,
        "crackling_fire":7,
        "helicopter":8,
        "chainsaw":9,
    }

    with open("../data/audio/ESC-50-master/meta/esc50.csv") as f:
        lines = [i[:-1] for i in f.readlines()]
    lines = lines[1:]

    os.system("rm -rf ../data/audio/ESC-10")
    os.system("mkdir ../data/audio/ESC-10")
    os.system("mkdir ../data/audio/ESC-10/audio")

    meta = []
    for line in lines:
        t = line.split(",")
        if (t[-3] == 'True'):
            meta.append("../data/audio/ESC-10/audio/%s %d" % (t[0],classes[t[3]]))
            src = "../data/audio/ESC-50-master/audio/"+t[0]
            dst = "../data/audio/ESC-10/audio/"+t[0]
            shutil.copy(src,dst)

    with open("../data/audio/ESC-10/filelist.txt","w") as f:
        for m in meta:
            f.write(m+"\n")


main()

