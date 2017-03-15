from __future__ import print_function
import argparse
import os

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions

from models.qrnn import QRNNLayer
from util.datasets import IMDBDataset
import util.util as util


class QRNNModel(chainer.Chain):
    def __init__(self, vocab_size, out_size, hidden_size, dropout):
        super().__init__(
            layer1=QRNNLayer(out_size, hidden_size),
            layer2=QRNNLayer(hidden_size, hidden_size),
            layer3=QRNNLayer(hidden_size, hidden_size),
            layer4=QRNNLayer(hidden_size, hidden_size),
            fc=L.Linear(None, 2)
        )
        self.embed = L.EmbedID(vocab_size, out_size)
        self.dropout = dropout
        self.train = True

    def __call__(self, x):
        h = self.embed(x)
        h = F.dropout(self.layer1(h), self.dropout, self.train)
        h = F.dropout(self.layer2(h), self.dropout, self.train)
        h = F.dropout(self.layer3(h), self.dropout, self.train)
        h = F.dropout(self.layer4(h), self.dropout, self.train)
        return self.fc(h)


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super().evaluate()
        model.train = True
        return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=24,
                        help='Number of documents in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--embeddings', default='',
                        help='Iinitial word embeddings file')
    parser.add_argument('--vocabulary', default='',
                        help='Vocabulary file')
    parser.add_argument('--dataset', default='data/aclImdb', type=str,
                        help='IMDB dataset path, dir with train/ and test/ folders')
    parser.add_argument('--vocab_size', default=68379, type=int,
                        help='GloVe word embedding dimensions')
    parser.add_argument('--out_size', default=300, type=int,
                        help='GloVe word embedding dimensions')
    parser.add_argument('--hidden_size', default=256, type=int,
                        help='Hidden layers dimensions')
    parser.add_argument('--maxlen', default=400, type=int,
                        help='Maximum sequence time (T) length')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout ratio between layers')
    return parser.parse_args()


def main():
    args = parse_args()

    train, test = IMDBDataset(os.path.join(args.dataset, 'train'), args.vocabulary, args.maxlen),\
                  IMDBDataset(os.path.join(args.dataset, 'test'), args.vocabulary, args.maxlen)


    model = L.Classifier(QRNNModel(
        args.vocab_size, args.out_size, args.hidden_size, args.dropout))

    if args.embeddings:
        model.predictor.embed.W.data = util.load_embeddings(
            args.embeddings, args.vocab_size, args.out_size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        model.predictor.embed.to_gpu()

    optimizer = chainer.optimizers.RMSprop(lr=0.001, alpha=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
