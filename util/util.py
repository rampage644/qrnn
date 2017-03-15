from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import numpy as np
import pickle
import re


def load_embeddings(path, size, dimensions):
    ret = np.zeros((size, dimensions), dtype=np.float32)

    size = os.stat(path).st_size
    with open(path, 'rb') as ifile:
        pos = 0
        idx = 1
        while pos < size:
            chunk = np.load(ifile)
            ret[idx:idx+chunk.shape[0], :] = chunk
            pos = ifile.tell()
    return ret


#%%
# path = 'data/embeddings.npy'

# r = load_embeddings(path)
# r[23]
# #%%

# dictpath = 'data/dict.pckl'
# with open(dictpath, 'rb') as dfile:
#     wdict = pickle.load(dfile)

# v1 = r[wdict['king']]
# v2 = r[wdict['queen']]
# v3 = r[wdict['man']]
# v4 = r[wdict['woman']]


# r1 = v1 - v3
# r2 = v2 - v4

# (r2 - r1)
