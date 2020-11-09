from math import sqrt
def advanced_metrics(tally, m): 
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

