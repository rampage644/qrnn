# QRNN

Simple sentiment analysis experiment with Quasi-recurrent neural networks (QRNN) and IMDb dataset. Implemented with `chainer`. More detailed explanation could be found in my blog post: http://sergeiturukin.com/2017/03/15/qrnn-imdb.html

## Dataset

IMDb dataset could be downloaded here: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
300-dimensional cased GloVe embeddings could be found here: http://nlp.stanford.edu/data/glove.6B.zip

## Instructions

Clone repo, download dataset and embeddings:

    git clone https://github.com/rampage644/qrnn
    cd qrnn
    mkdir data/
    cd data/
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    wget http://nlp.stanford.edu/data/glove.6B.zip
    tar xfz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    unzip http://nlp.stanford.edu/data/glove.6B.zip

Preprocess the data (GloVe embeddings):

    python3 glove_to_npy.py -d data/glove.840B.300d.txt --npy_output data/embeddings.npy --dict_output data/vocab.pckl --dict_whitelist data/aclImdb/imdb.vocab

Model relies on modified `chainer` version. Install it from sources https://github.com/jekbradbury/chainer/tree/raw-kernel. In case of problems follow my blog post (see link in above) to apply patch to current version of `chainer` (though you still need to install it from sources).

Train the model with default parameters:

    python3 train.py -g0 -o results/ --dataset data/aclImdb/ --vocabulary data/vocab.pckl --embeddings data/embeddings.npy

## Results

With default parameters (only dropout set to 0.5) model achieves ~86% accuracy on validatio set. Hyperparameters tuning was not performed. Model suffers heavily from overfitting. 20 epoch on single NVIDIA Titan X Pascal GPU take ~10 minutes.

    $ python3 train.py -g0 -o results/ --dataset data/aclImdb/ --vocabulary data/vocab.pckl --embeddings data/embeddings.npy --dropout 0.5
    epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy
    1           0.542545    0.415724              0.748241       0.815159
    2           0.412443    0.371796              0.819378       0.838072
    3           0.37779     0.348942              0.836295       0.854506
    4           0.354687    0.470325              0.849848       0.800603
    5           0.337171    0.336974              0.857885       0.863385
    6           0.317884    0.39081               0.865153       0.839712
    7           0.304117    0.350891              0.872921       0.853087
    8           0.291776    0.330035              0.880759       0.860645
    9           0.277024    0.365185              0.887127       0.854467
    10          0.263624    0.404913              0.893514       0.835333
    11          0.248957    0.365441              0.898472       0.862405
    12          0.233838    0.383892              0.908221       0.856066
    13          0.221411    0.380842              0.912828       0.860224
    14          0.211432    0.423715              0.915187       0.848769
    15          0.194838    0.442775              0.92195        0.851088
    16          0.184106    0.469984              0.928222       0.851787
    17          0.175725    0.472484              0.931701       0.861365
    18          0.164538    0.518889              0.934638       0.849648
    19          0.157166    0.46597               0.939738       0.850867
    20          0.147039    0.549008              0.942138       0.858005
