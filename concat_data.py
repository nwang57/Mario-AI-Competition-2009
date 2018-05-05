import numpy as np
import sys

f1 = sys.argv[1]
f2 = sys.argv[2]

n1 = np.load(f1)
n2 = np.load(f2)

n_cat = np.concatenate((n1, n2[1:]), axis=0)

np.save(sys.argv[3], n_cat)


