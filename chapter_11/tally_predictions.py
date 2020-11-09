def tally_predictions(clf, x, y): 
    p = clf.predict(x)
    score = clf.score(x,y)
    tp = tn = fp = fn = 0 
    for i in range(len(y)):
        if (p[i] == 0) and (y[i] == 0): 
            tn += 1
        elif (p[i] == 0) and (y[i] == 1): 
            fn += 1
        elif (p[i] == 1) and (y[i] == 0): 
            fp += 1
        else:
            tp += 1
    return [tp, tn, fp, fn, score]

