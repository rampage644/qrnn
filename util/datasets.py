'''Preprocess IMDB dataset.'''
from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
import os
import argparse
import pickle

import numpy as np

from chainer.dataset import dataset_mixin

def convert_file(filepath, word_dict):
    with open(filepath) as ifile:
        return [word_dict.get(w, 0) for w in ifile.read().split(' ')]


def discover_dataset(path, wdict):
    dataset = []
    for root, _, files in os.walk(path):
        for sfile in [f for f in files if '.txt' in f]:
            filepath = os.path.join(root, sfile)
            dataset.append(convert_file(filepath, wdict))
    return dataset


def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen-len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen])
         for r in dataset])

#%%
class IMDBDataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, dict_path, maxlen=128):
        pos_path = os.path.join(path, 'pos')
        neg_path = os.path.join(path, 'neg')

        with open(dict_path, 'rb') as dfile:
            wdict = pickle.load(dfile)

        self.pos_dataset = pad_dataset(discover_dataset(pos_path, wdict), maxlen).astype('i')
        self.neg_dataset = pad_dataset(discover_dataset(neg_path, wdict), maxlen).astype('i')

    def __len__(self):
        return len(self.pos_dataset) + len(self.neg_dataset)

    def get_example(self, i):
        is_neg = i >= len(self.pos_dataset)
        dataset = self.neg_dataset if is_neg else self.pos_dataset
        idx = i - len(self.pos_dataset) if is_neg else i
        label = 0 if is_neg else 1

        return (dataset[idx], np.array(label, dtype=np.int32))

#%%
# import chainer
# train = IMDBDataset('data/aclImdb/train', 'data/dict.pckl')
# train_iter = chainer.iterators.SerialIterator(train, 16)


# batch = next(train_iter)
# batch[0]

# chainer.dataset.concat_examples(batch)
