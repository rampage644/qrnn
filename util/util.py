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
        idx = 0
        while pos < size:
            chunk = np.load(ifile)
            chunk_size = chunk.shape[0]
            ret[idx:idx+chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return ret
