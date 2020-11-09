#
#  file:  mnist_cnn_fcn_graymaps.py
#
#  Generate per digit grayscale heat maps
#  for the given input image.
#
#  RTK, 28-Oct-2019
#  Last update:  28-Oct-2019
#
################################################################

import os
import sys
import numpy as np
from PIL import Image

def main():
    if (len(sys.argv) != 5):
        print()
        print("mnist_cnn_fcn_graymaps <threshold> <image> <results> <outdir>")
        print()
        print("  <threshold> - heatmap threshold [0,1)")
        print("  <image>     - source image")
        print("  <results>   - results for image (.npy)")
        print("  <outdir>    - output graymap directory (overwritten)")
        print()
        return

    threshold = float(sys.argv[1])
    iname = sys.argv[2]
    rname = sys.argv[3]
    outdir= sys.argv[4]

    os.system("rm -rf %s; mkdir %s" % (outdir, outdir))

    res = np.load(rname)
    img = Image.open(iname)
    c,r = img.size


    #  Process all large images
    inames = ["images/"+i for i in os.listdir("images")]
    rnames = ["results/"+i for i in os.listdir("results")]
    inames.sort()
    rnames.sort()

    hmap = np.zeros((r,c,10))
    res = np.load(rname)
    x,y,_ = res.shape
    xoff = (r - 2*x) // 2
    yoff = (c - 2*y) // 2

    for j in range(10):
        h = np.array(Image.fromarray(res[:,:,j]).resize((2*y,2*x)))
        hmap[xoff:(xoff+x*2), yoff:(yoff+y*2),j] = h 

    #  Store the raw heatmap
    np.save("%s/graymaps.npy" % outdir, hmap)
    
    #  Apply the threshold
    hmap[np.where(hmap < threshold)] = 0.0

    #  Convert heatmaps to grayscale images
    for j in range(10):
        img = np.zeros((r,c), dtype="uint8")
        for x in range(r):
            for y in range(c):
                img[x,y] = int(255.0*hmap[x,y,j])

        Image.fromarray(img).save("%s/graymap_digit_%d.png" % (outdir, j))
        img = 255-img
        Image.fromarray(img).save("%s/graymap_inv_digit_%d.png" % (outdir, j))



main()

