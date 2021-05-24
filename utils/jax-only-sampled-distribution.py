from os import listdir, path, rename

import json
import numpy as np
import pdal
import random

rootdir = '/home/chambbj/data/ml-datasets/US3D/oma-only'
counts = {0:0,2:0,5:0,6:0,7:0,9:0,17:0}
for filename in listdir(path.join(rootdir, "validation")):
    p = pdal.Pipeline(json.dumps([path.join(rootdir, "input_2.000", filename)]))
    p.execute()
    pc = p.arrays[0]

    classes = pc['Classification']
    unique_classes = np.unique(classes)
    bins = np.bincount(classes)

    for c in unique_classes:
        counts[c] += bins[c]

    print(counts)