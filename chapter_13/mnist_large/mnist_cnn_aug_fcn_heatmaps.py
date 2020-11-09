#
#  file:  mnist_cnn_fcn_heatmaps.py
#
#  Generate per digit heatmaps
#
#  RTK, 20-Oct-2019
#  Last update:  20-Oct-2019
#
################################################################

import os
import sys
import numpy as np
from PIL import Image

def main():
    if (len(sys.argv) > 1):
        threshold = float(sys.argv[1])
    else:
        threshold = 0.99

    os.system("rm -rf heatmaps_aug; mkdir heatmaps_aug")

    #  Process all large images
    inames = ["images/"+i for i in os.listdir("images")]
    rnames = ["results_aug/"+i for i in os.listdir("results_aug")]
    inames.sort()
    rnames.sort()

    #  Heat map colors
    colors = [
        [0xe6,0x19,0x4b],[0xf5,0x82,0x31],[0xff,0xe1,0x19],[0xbf,0xef,0x45],[0x3c,0xb4,0x4b],
        [0x42,0xd4,0xf4],[0x43,0x63,0xd8],[0x91,0x1e,0xb4],[0xf0,0x32,0xe6],[0xa9,0xa9,0xa9]
    ]

    for i,iname in enumerate(inames):
        rname = rnames[i]
        c,r = Image.open(iname).size
        m = np.load(rname)
        hmap = np.zeros((r,c,10))
        res = np.load(rname)
        x,y,_ = res.shape
        xoff = (r - 2*x) // 2
        yoff = (c - 2*y) // 2

        #  Create heatmaps for this input image
        for j in range(10):
            h = np.array(Image.fromarray(res[:,:,j]).resize((2*y,2*x)))
            hmap[xoff:(xoff+x*2), yoff:(yoff+y*2),j] = h 

        #  Store the raw heatmap
        np.save("heatmaps_aug/heatmap_%04d.npy" % i, hmap)
        
        #  Apply the threshold
        hmap[np.where(hmap < threshold)] = 0.0

        #  Convert heatmaps to RGB image
        cmap = np.zeros((r,c,3), dtype="uint8")
        for x in range(r):
            for y in range(c):
                if (hmap[x,y,:].max() >= threshold):
                    n = np.argmax(hmap[x,y,:])
                    cmap[x,y,:] = colors[n]

        #  Store the image as well
        cmap = Image.fromarray(cmap)
        cmap.save("heatmaps_aug/heatmap_%04d.png" % i)

        #  Alpha-blend
        img = Image.blend(cmap.convert("RGBA"), Image.open(iname).convert("RGBA"), alpha=0.2)
        img.convert("RGB").save("heatmaps_aug/blend_%04d.png" % i)


main()

