from os import listdir, path, rename

import json
import numpy as np
import pdal
import random

rootdir = '/home/chambbj/data/ml-datasets/US3D/oma-only'
for filename in listdir(path.join(rootdir, "all")):
    p = pdal.Pipeline(json.dumps([path.join(rootdir, "all", filename)]))
    p.execute()
    pc = p.arrays[0]

    classes = pc['Classification']
    unique_classes = np.unique(classes)

    if any(x in [7, 9, 17] for x in unique_classes):
        r = random.random()
        print("minority", r)
        if (r < 0.2):
            rename(path.join(rootdir, "all", filename), path.join(rootdir, "validation", filename))
        else:
            rename(path.join(rootdir, "all", filename), path.join(rootdir, "train", filename))
    else:
        r = random.random()
        print("majority", r)
        if (r < 0.2):
            rename(path.join(rootdir, "all", filename), path.join(rootdir, "validation", filename))
        else:
            rename(path.join(rootdir, "all", filename), path.join(rootdir, "train", filename))