import numpy as np

f = [0.3408,3.0150,4.3465,2.1271,2.7561,
     2.7887,4.8231,0.0705,3.9897,0.9804,
     2.3944,2.0085,1.7821,1.5362,2.3190]
f = np.array(f)

print
print("mean  = %0.4f" % f.mean())
print("std   = %0.4f" % f.std())
print("SE    = %0.4f" % (f.std()/np.sqrt(f.shape[0])))
print("median= %0.4f" % np.median(f))
print("min   = %0.4f" % f.min())
print("max   = %0.4f" % f.max())

