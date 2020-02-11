#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import sys
sys.path.insert(0,'../..')

import IPython
from unittest import TestCase, main, skip
import tempfile
import numpy as np
import torchfit

# torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

# torchtext for preprocessing
import torchtext
from torchtext.datasets import text_classification

SEED = 42
DEVICE = 'cuda'

class Net(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, data):
        text = data[0]
        offsets = data[1]
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
def accuracy(y_true, y_pred):
    return np.mean(y_true.numpy() == np.argmax(y_pred.numpy(), axis=1))
    

class TestTorchFit(TestCase):

    #@skip('temporarily disabled')
    def test_torchfit(self):
        # DATALOADERS
        NGRAMS = 2
        import os
        if not os.path.isdir('./.data'):
                os.mkdir('./.data')
        train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
            root='./.data', ngrams=NGRAMS, vocab=None)
        BATCH_SIZE = 16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def generate_batch(batch):
            label = torch.tensor([entry[0] for entry in batch])
            text = [entry[1] for entry in batch]
            offsets = [0] + [len(entry) for entry in text]
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            text = torch.cat(text)
            return [text, offsets], label

        train_len = int(len(train_dataset) * 0.95)
        sub_train_, sub_valid_ = \
            random_split(train_dataset, [train_len, len(train_dataset) - train_len])
        train_loader = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=generate_batch)
        val_loader = DataLoader(sub_valid_, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=generate_batch)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=generate_batch)
        trn = train_loader
        val = val_loader
        tst = test_loader

        VOCAB_SIZE = len(train_dataset.get_vocab())
        EMBED_DIM = 32
        NUM_CLASS = len(train_dataset.get_labels())


        lr = 4.0
        model = Net(VOCAB_SIZE, EMBED_DIM, NUM_CLASS)

        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        criterion = torch.nn.CrossEntropyLoss()
        learner = torchfit.Learner(model, trn, val_loader=val, 
                                   optimizer=optimizer, loss=criterion, metrics=[accuracy], 
                                   seed=SEED, device=DEVICE)

        hist = learner.fit_onecycle(lr, 1)
        self.assertLess(min(hist.get_epoch_log('val_loss')), 0.3500)

        self.assertAlmostEqual(max(hist.get_batch_log('lrs')), lr)


        outputs, targets = learner.predict(tst, return_targets=True)
        outputs = np.argmax(outputs, axis=-1)
        self.assertGreater(np.mean(targets == outputs), 0.88)

        new_file, tmpfile = tempfile.mkstemp()
        learner.save(tmpfile)


        model = Net(VOCAB_SIZE, EMBED_DIM, NUM_CLASS)
        learner = torchfit.Learner(model)
        learner.load(tmpfile)
        outputs, targets = learner.predict(tst, return_targets=True)
        outputs = np.argmax(outputs, axis=-1)
        self.assertGreater(np.mean(targets == outputs), 0.88)


        import re
        from torchtext.data.utils import ngrams_iterator
        from torchtext.data.utils import get_tokenizer
        vocab = train_dataset.get_vocab()

        labels = ['World', 'Sports', 'Business', 'Sci/Tech']

        def preprocess(text):
            tokenizer = get_tokenizer("basic_english")
            with torch.no_grad():
                text = torch.tensor([vocab[token]
                                    for token in ngrams_iterator(tokenizer(text), 2)])
                return [text, torch.tensor([0])]

        text = 'The stock price of IBM shot up today after its earnings report.'
        pred = learner.predict_example(text, preproc_fn=preprocess, labels=labels)
        self.assertEqual(pred, 'Business')


if __name__ == "__main__":
    main()
