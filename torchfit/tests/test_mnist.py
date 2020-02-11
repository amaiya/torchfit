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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
SEED = 42
DEVICE = 'cuda'
BATCH_SIZE = 32

# define a PyTorch model as you normally would
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
def accuracy(y_true, y_pred):
    return np.mean(y_true.numpy() == np.argmax(y_pred.numpy(), axis=1))
    

class TestMNIST(TestCase):

    def setUp(self):


        # setup DataLoaders for training and validation as you normally would
        kwargs = {'num_workers': 1, 'pin_memory': True} if 'cuda' in DEVICE else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=BATCH_SIZE, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/mnist', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=BATCH_SIZE, shuffle=False, **kwargs)
        self.trn = train_loader
        self.val = test_loader


    #@skip('temporarily disabled')
    def test_mnist(self):
        lr = 1e-3
        epochs = 1
        dummy_lr = 100
        model = Net()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        learner = torchfit.Learner(model, self.trn, val_loader=self.val, 
                                   optimizer=optimizer, criterion=criterion, metrics=[accuracy], 
                                   seed=SEED, device=DEVICE)
        scheduler = torch.optim.lr_scheduler.CyclicLR(learner.optimizer, 
                                                     base_lr=lr/10, 
                                                     max_lr=lr, 
                                                     cycle_momentum=False, 
                                                     step_size_up=epochs*len(self.trn)//3,
                                                     step_size_down=epochs*2*len(self.trn)//3)
        hist = learner.fit(dummy_lr, epochs, schedulers=[scheduler])

        self.assertLess(min(hist.get_epoch_log('val_loss')), 0.1400)

        self.assertAlmostEqual(max(hist.get_batch_log('lrs')), lr)


        outputs, targets = learner.predict(self.val, return_targets=True)
        outputs = np.argmax(outputs, axis=-1)
        self.assertGreater(np.mean(targets == outputs), 0.94)

        new_file, tmpfile = tempfile.mkstemp()
        learner.save(tmpfile)


        model = Net()
        learner = torchfit.Learner(model)
        learner.load(tmpfile)
        outputs, targets = learner.predict(self.val, return_targets=True)
        outputs = np.argmax(outputs, axis=-1)
        self.assertGreater(np.mean(targets == outputs), 0.94)



if __name__ == "__main__":
    main()
