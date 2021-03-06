'''Convert pretrained GloVe embeddings into npy file'''
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pickle
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--npy_output', type=str, required=True)
    parser.add_argument('--dict_output', type=str, required=True)
    parser.add_argument('--dict_whitelist', type=str, required=True)
    parser.add_argument('--dump_frequency', type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()

    data = {
        '': 0
    }
    embeddings = [
        np.zeros((300), dtype=np.float32)
    ]

    float_re = re.compile(' [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')
    # float_re = re.compile(' -?\d+\.\d+')

    with open(args.dict_whitelist) as wfile:
        whitelist = [line.strip() for line in wfile]

    with open(args.dataset) as ofile, \
         open(args.dict_output, 'wb') as dfile, \
         open(args.npy_output, 'wb') as nfile:
        idx = 1
        for line in ofile:
            pos = next(re.finditer(float_re, line)).start()
            word, vector = line[:pos], line[pos+1:].split()

            if word not in whitelist:
                continue

            if word in data:
                print('Possible duplicate at {} in {}'.format(idx, line))
                continue
            embedding = np.fromiter([float(d) for d in vector], np.float32)
            if embedding.shape != (300,):
                print('Shape is {}'.format(embedding.shape))
                print(line)
            embeddings.append(embedding)
            data[word] = idx

            idx += 1
            if not idx % args.dump_frequency:
                np.save(nfile, np.array(embeddings))
                embeddings.clear()

        np.save(nfile, np.array(embeddings))
        pickle.dump(data, dfile)


if __name__ == '__main__':
    main()
