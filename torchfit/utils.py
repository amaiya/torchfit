import os
import sys
from collections import OrderedDict

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np



class History():
    def __init__(self):
        self.epoch_log = []
        self.batch_log = {}
    def update_epoch_log(self, log):
        if type(log) not in [OrderedDict]:
            raise ValueError('log must be an OrderedDict')
        self.epoch_log.append(log)
        return
    def update_batch_log(self, key, value):
        lst = self.batch_log.get(key, [])
        lst.append(value)
        self.batch_log[key] = lst
        return
    def get_epoch_log(self, key):
        results = []
        for d in self.epoch_log:
            value = d.get(key, None)
            if value is None: raise ValueError('could not find key %s' % (key))
            results.append(value)
        return results
    def get_batch_log(self, key):
        values = self.batch_log.get(key, None)
        if values is None: raise ValueError('could not find key %s' % (key))
        return values
    def has_epoch_key(self, key):
        return self.epoch_log and key in self.epoch_log[0]
    def has_batch_key(self, key):
        return key in self.batch_log


def show_image(img_path):
    if not os.path.isfile(img_path):
        raise ValueError('%s is not valid file' % (img_path))
    img = plt.imread(img_path)
    out = plt.imshow(img)
    return out



def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

#-------------------------------------------------------------------------------
# The ProgressBar, log_to_message, and add_metrics_to_log functions
# support Keras-style training progress in Jupyter notebooks.
# Code Reference:  https://github.com/henryre/pytorch-fitmodule
#-------------------------------------------------------------------------------
def add_metrics_to_log(log, metrics, y_true, y_pred, prefix=''):
    for metric in metrics:
        q = metric(y_true, y_pred)
        log[prefix + metric.__name__] = q
    return log


def log_to_message(log, precision=4, omit=[]):
    fmt = "{0}: {1:." + str(precision) + "f}"
    sep = "  "
    return sep+sep.join(fmt.format(k, v) for k, v in log.items() if k not in omit)


class ProgressBar(object):
    """Cheers @ajratner"""

    def __init__(self, n, length=40):
        # Protect against division by zero
        self.n      = max(1, n)
        self.nf     = float(n)
        self.length = length
        # Precalculate the i values that should trigger a write operation
        self.ticks = set([round(i/100.0 * n) for i in range(101)])
        self.ticks.add(n-1)
        self.bar(0)

    def bar(self, i, message=""):
        """Assumes i ranges through [0, n-1]"""
        if i in self.ticks:
            b = int(np.ceil(((i+1) / self.nf) * self.length))
            sys.stdout.write("\r[{0}{1}] {2}%\t{3}".format(
                "="*b, " "*(self.length-b), int(100*((i+1) / self.nf)), message
            ))
            sys.stdout.flush()

    def close(self, message=""):
        # Move the bar to 100% before closing
        self.bar(self.n-1)
        sys.stdout.write("{0}\n\n".format(message))
        sys.stdout.flush()
