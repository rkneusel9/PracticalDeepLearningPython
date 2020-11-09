#
#  file:  make_roc_points_sklearn.py
#
#  Generate the ROC curve points for the
#  given set of output probabilities.
#
#  RTK, 18-Apr-2019
#  Last update:  21-Apr-2019
#
###############################################################

import os
import sys
import numpy as np
import matplotlib.pylab as plt

from sklearn.metrics import roc_auc_score, roc_curve

def main():
    if (len(sys.argv) == 1):
        print()
        print("make_roc_points_sklearn <labels> <probs> <plot>")
        print()
        print("  <labels>  -  test labels (.npy)")
        print("  <probs>   -  test per class probabilities (.npy)")
        print("  <plot>    -  output ROC plot (.png)")
        print()
        return

    labels = np.load(sys.argv[1])
    probs = np.load(sys.argv[2])
    pname = sys.argv[3]

    auc = roc_auc_score(labels, probs[:,1])
    roc = roc_curve(labels, probs[:,1])
    print("AUC = %0.6f" % auc)
    print()

    plt.plot(roc[0], roc[1], color='r')
    plt.plot([0,1],[0,1], color='k', linestyle=':')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(pname, dpi=300)
    plt.show()


main()

