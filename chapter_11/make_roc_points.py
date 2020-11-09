#
#  file:  make_roc_points.py
#
#  Generate the ROC curve points for the
#  given set of output probabilities.
#
#  RTK, 18-Apr-2019
#  Last update:  20-Apr-2019
#
###############################################################

import os
import sys
import numpy as np
import matplotlib.pylab as plt

from sklearn.metrics import roc_auc_score

def basic_metrics(tally):
    """Use the tallies to generate basic metrics"""

    tp, tn, fp, fn, _ = tally
    return {
        "TPR": tp / (tp + fn),
        "TNR": tn / (tn + fp),
        "PPV": tp / (tp + fp),
        "NPV": tn / (tn + fn),
        "FPR": fp / (fp + tn),
        "FNR": fn / (fn + tp)
    }


from math import sqrt
def advanced_metrics(tally, m):
    """Use the tallies to calculate more advanced metrics"""

    tp, tn, fp, fn, _ = tally
    n = tp+tn+fp+fn

    po = (tp+tn)/n
    pe = (tp+fn)*(tp+fp)/n**2 + (tn+fp)*(tn+fn)/n**2

    return {
        "F1": 2.0*m["PPV"]*m["TPR"] / (m["PPV"] + m["TPR"]),
        "MCC": (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        "kappa": (po - pe) / (1.0 - pe),
        "informedness": m["TPR"] + m["TNR"] - 1.0,
        "markedness": m["PPV"] + m["NPV"] - 1.0
    }


def table(labels, probs, t):
    """Return a 2x2 table using the given threshold"""

    tp = tn = fp = fn = 0
    
    for i,l in enumerate(labels):
        c = 1 if (probs[i,1] >= t) else 0
        if (l == 0) and (c == 0):
            tn += 1
        if (l == 0) and (c == 1):
            fp += 1
        if (l == 1) and (c == 0):
            fn += 1
        if (l == 1) and (c == 1):
            tp += 1

    return [tp, tn, fp, fn]


def main():
    if (len(sys.argv) == 1):
        print()
        print("make_roc_points <labels> <probs> <points> [<plot>]")
        print()
        print("  <labels>  -  test labels (.npy)")
        print("  <probs>   -  test per class probabilities (.npy)")
        print("  <points>  -  output ROC points (.npy)")
        print("  <plot>    -  output ROC plot (.png)")
        print()
        return

    labels = np.load(sys.argv[1])
    probs = np.load(sys.argv[2])
    oname = sys.argv[3]
    pname = "" if (len(sys.argv) < 5) else sys.argv[4]

    th = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    roc = []

    for t in th:
        tp, tn, fp, fn = table(labels, probs, t)
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        roc.append([fpr, tpr])

        tally = [tp, tn, fp, fn, 1]
        m = basic_metrics(tally)
        adv = advanced_metrics(tally, m)
        print("theta = %0.1f:" % t)
        print("    MCC = %0.5f, AUC = %0.5f" % (adv["MCC"], roc_auc_score(labels, probs[:,1])))
        print("    TPR = %0.5f, FPR = %0.5f, PPV = %0.5f" % (m["TPR"], m["FPR"], m["PPV"]))

    roc = np.array(roc)
    np.save(oname, roc)
    print()
    print("ROC points:")
    print(roc)
    print()


    if (pname != ""):
        xy = np.zeros((roc.shape[0]+2, roc.shape[1]))
        xy[1:-1,:] = roc
        xy[0,:] = [0,0]
        xy[-1,:] = [1,1]
        plt.plot(xy[:,0], xy[:,1], color='r', marker='o')
        plt.plot([0,1],[0,1], color='k', linestyle=':')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.savefig(pname, dpi=300)
        plt.show()


main()

