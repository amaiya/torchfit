import torch
from torch.nn import CrossEntropyLoss, Module
from torch import optim

import math
from collections import OrderedDict
from functools import partial
import tempfile
import warnings

import numpy as np
from matplotlib import pyplot as plt
from .utils import add_metrics_to_log, log_to_message, ProgressBar, History


DEFAULT_LOSS = CrossEntropyLoss()
DEFAULT_OPTIMIZER = partial(optim.SGD, lr=0.001, momentum=0.9)
def accuracy(y_true, y_pred):
    return np.mean(y_true.numpy() == np.argmax(y_pred.numpy(), axis=1))
DEFAULT_METRIC = accuracy

SCHED_LOCATIONS = { 'LambdaLR': 'epochlevel',
                    'MultiplicativeLR': 'epochlevel',
                    'StepLR': 'epochlevel',
                    'MultiStepLR': 'epochlevel',
                    'ExponentialLR': 'epochlevel',
                    'CosineAnnealingLR': 'epochlevel',
                    'ReduceLROnPlateau': 'epochlevel',
                    'CyclicLR': 'batchlevel',
                    'CosineAnnealingWarmRestarts': 'batchlevel',
                    'OneCycleLR': 'batchlevel', }



class Learner():

    
    def __init__(self, model, train_loader, val_loader=None,
                 loss=None,        # default is crossentropy
                 optimizer=None,    # default sgd
                 metrics=[accuracy], # default is accuracy
                 seed=None,
                 device=None,
                 use_amp=False,
                 amp_level='O1',
                 verbose=1):
        """
        Learner constructor

        Args:
          model (nn.Module): the network
          train_loader(DataLoader):  training dataset
          val_loader(DataLoader):  validation dataset
          loss(nn._Loss): loss function. Default is CrossEntropy if None.
          optimizer(Optimizer):  optimizer.  Default is SGD if None.
          metrics(list): list of functions for computing metrics. Default is accuracy.
          seed(int): random seed
          device(str): cuda or cpu.  If None, inferred automatically.
          use_amp(bool): train using automatic mixed precision.  default:False
          amp_level(str): opt_level for automatic mixed precision.  default:'O1'
                          https://nvidia.github.io/apex/amp.html
          verbose(int): verbosity
        """
        # set seed
        if seed and seed >= 0:
            torch.manual_seed(seed)

        # set device
        if device is None:
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # instance variables
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.verbose = verbose
        self.loss = loss
        if self.loss is None: self.loss = DEFAULT_LOSS
        self.optimizer = optimizer
        if self.optimizer is None:
            opt = DEFAULT_OPTIMIZER
            self.optimizer = opt(self.model.parameters())
        self.metrics = metrics
        if self.metrics and not isinstance(self.metrics, list):
            raise ValueError('metrics must be a list of functions')
        self.hist = None

        # save state
        self.state = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),}

        # amp
        self.use_amp = use_amp
        if self.use_amp:
            try:
                from apex import amp
            except ImportError:
                msg = """
                You set `use_amp=True` but do not have apex installed.
                Install apex first using this guide and rerun with use_amp=True:
                https://github.com/NVIDIA/apex#linux

                mixed precision training will NOT be used for this run
                """
                self.use_amp = False
                warnings.warn(msg)
        if self.use_amp:
            print("using mixed precision training (EXPERIMENTAL)")
            try:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=amp_level)
                print('here')
            except RuntimeError as e:
                msg = """
                You may have already amp.initialized this model and optimizer.
                Re-instantiate your model and optimizer.
                """
                raise RuntimeError(str(e)+'\n\n'+msg)
            self.amp_init=True



    def fit(self, lr, epochs=1, 
            schedulers=None, accumulation_steps=1):
        """
        train the model
        Args:
          lr (float):  learning rate
          epochs (int):  number of epochs.  default:1
          schedulers(list):  list of LR schedulers.  Default is None.
          accumulation_steps(int): number of batches for gradient accumulation.
                                   default:1
        Returns:
          History:  History object containing training history
        """
        # history
        self.hist = History()

        # check amp
        if self.use_amp:
            from apex import amp

        # check schedulers
        schedulers = schedulers
        if schedulers is not None:
            if type(schedulers) != list: raise ValueError('schedulers must be list of _LRScheduler instances')
            unk_schedulers = []
            for s in schedulers:
                if type(s).__name__ not in SCHED_LOCATIONS:
                    unk_schedulers.append(type(s).__name__)
            if len(unk_schedulers) > 0:
                raise ValueError('unknown schedulers  were supplied: %s'  % (' '.join(unk_schedulers)))
        

        # set learning rate        
        opt = self.optimizer
        for g in opt.param_groups:
            g['lr'] = lr

        # training loop
        logs = []
        for t in range(epochs):
            if self.verbose:
                print("Epoch {0} / {1}".format(t+1, epochs))

            # setup logger
            if self.verbose:
                pb = ProgressBar(len(self.train_loader))
            log = OrderedDict()
            epoch_loss = 0.0

            # run batches for this epoch
            for batch_i, batch_data in enumerate(self.train_loader):
                self.model.train()

                # get batch
                X_batch = batch_data[0].to(self.device) 
                y_batch = batch_data[1].to(self.device)

                # forward and backward pass
                y_batch_pred = self.model(X_batch)
                batch_loss = self.loss(y_batch_pred, y_batch)
                if self.use_amp:
                    with amp.scale_loss(batch_loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    batch_loss.backward()
                #batch_loss.backward()

                if (batch_i+1) % accumulation_steps == 0:
                    opt.step()
                    opt.zero_grad()

                    # update status
                    epoch_loss += batch_loss.item()
                    log['loss'] = float(epoch_loss) / (batch_i + 1)
                    if self.verbose:
                        pb.bar(batch_i, log_to_message(log))


                    # lr schedule
                    if schedulers:
                        for s in schedulers:
                            if SCHED_LOCATIONS[type(s).__name__] == 'batchlevel': s.step()

                    # update lr log
                    if schedulers:
                        try:
                            last_lr = schedulers[0].get_last_lr()
                        except:
                           last_lr = schedulers[0].get_lr()
                           #last_lr = self.optimizer.param_groups[0]['lr']
                        self.hist.update_batch_log('lrs', last_lr)


            # Run metrics
            if self.metrics:
                y_train_pred, train_loss, y_train = self._predict(self.train_loader)
                add_metrics_to_log(log, self.metrics, y_train, y_train_pred)
            if self.val_loader is not None:
                y_val_pred, val_loss, y_val = self._predict(self.val_loader)
                log['val_loss'] = val_loss
                if self.metrics:
                    add_metrics_to_log(log, self.metrics, y_val, y_val_pred, 'val_')

            # update status
            logs.append(log)
            if self.verbose:
                pb.close(log_to_message(log, omit=['loss']))
            self.hist.update_epoch_log(log)

            # LR schedule
            if schedulers:
                for s in schedulers:
                    if SCHED_LOCATIONS[type(s).__name__] != 'batchlevel': s.step()
        return self.hist


    def fit_onecycle(self, lr, epochs=1, start_pct=0.25):
        """
        Train using Leslie Smith's 1cycle policy (https://arxiv.org/pdf/1803.09820.pdf)
        Args:
          max_lr(float):  maximum learning rate (lr for apex of triangle)
          epochs(int): number of epochs to train. Default is 1.
          start_pct(float): percentage of cycle that is increasing.
                            Using <0.5 slants triangle to left as proposed by Howard and 
                            Ruder (2018): https://arxiv.org/abs/1801.06146
                            Default is 0.25.
        """
        end_pct = 1-start_pct
        trn = self.train_loader
        scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                base_lr=lr/100, 
                                                max_lr=lr, 
                                                cycle_momentum=False, 
                                                step_size_up=math.floor(epochs*len(trn)*start_pct),
                                                step_size_down=math.floor(epochs*len(trn)*end_pct))
        return self.fit(lr, epochs, schedulers=[scheduler])



    def predict(self, data_loader):
        """Generates output predictions for the input samples.

        Args:
          data_loader(DataLoader): DataLoader instance
        Returns:      
          np.ndarray:  NumPy array of predictions
        """
        preds, _, _ = self._predict(data_loader)
        if self.device != 'cpu':
            preds = preds.cpu().detach()
        return preds.numpy()


    def _predict(self, data_loader):
        """
        Generates output predictions for the input samples
        along with returning loss and ground truth labels

        Args:
            data_loader: DataLoader instance
        Returns:
          (Tensor, float, Tensor): prediction Tensor, loss, ground truth Tensor
        """

        labeled = len(next(iter(data_loader))) > 1

        # Batch prediction
        r, n = 0, len(data_loader.dataset)
        total_loss = 0.0
        with torch.no_grad():
            for batch_i, batch_data in enumerate(data_loader):
                self.model.eval()
                # Predict on batch
                #X_batch = Variable(batch_data[0])
                X_batch = batch_data[0].to(self.device)
                y_batch = batch_data[1].to(self.device)
                y_batch_pred = self.model(X_batch).data

                # shapes
                pred_shape = (n,) if len(y_batch_pred.size()) == 1 \
                                  else (n,) + y_batch_pred.size()[1:]
                true_shape = (n,) if len(y_batch.size()) == 1 \
                                  else (n,) + y_batch.size()[1:]

                # loss
                batch_loss = self.loss(y_batch_pred, y_batch)
                total_loss += batch_loss.item()

                # Infer prediction shape
                if r == 0:
                    #y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
                    y_pred = torch.zeros(pred_shape)
                    y_true = torch.zeros(true_shape)
                # Add to prediction tensor
                y_pred[r : min(n, r + data_loader.batch_size)] = y_batch_pred
                y_true[r : min(n, r + data_loader.batch_size)] = y_batch
                r += data_loader.batch_size
        final_loss = float(total_loss) / (batch_i + 1)
        self.model.train()
        return y_pred, final_loss,  y_true


    def find_lr(self, 
                use_val_loss=False,
                start_lr=1e-6, 
                end_lr=10, 
                num_iter=100,
                step_mode="exp",
                smooth_f=0.05,
                diverge_th=5,
                accumulation_steps=1):
        """
        Finds a good learning rate.

        This method is simply a wrapper to torch-lr-finder module:
        https://github.com/davidtvs/pytorch-lr-finder

        Args:
            use_val_loss (bool): If False, the LR range test
                will only use the training loss. If True, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: False.
            start_lr (float, optional): the starting learning rate for the range test.
                Default: None (uses the learning rate from the optimizer).
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.
            accumulation_steps (int, optional): steps for gradient accumulation. If it
                is 1, gradients are not accumulated. Default: 1.
        """
        curr_opt_state = self.optimizer.state_dict()
        self.optimizer.load_state_dict(self.state['optimizer'])
        try:
            import logging
            logging.getLogger('torch_lr_finder').setLevel(logging.CRITICAL)
        except: pass
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from torch_lr_finder import LRFinder

        opt = self.optimizer

        if use_val_loss and self.val_loader is None:
            raise ValueError('use_val_loss=True but self.val_loader is None')
        val_loader = None
        if use_val_loss:
            val_loader = self.val_loader
        lr_finder = LRFinder(self.model, opt, self.loss, device=self.device)

        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            lr_finder.range_test(self.train_loader, 
                                 val_loader=val_loader,
                                 start_lr = start_lr,
                                 end_lr = end_lr,
                                 num_iter=num_iter,
                                 step_mode=step_mode,
                                 smooth_f=smooth_f,
                                 diverge_th=diverge_th,
                                 accumulation_steps=accumulation_steps)
        lr_finder.plot(log_lr=True)
        lr_finder.reset()
        print('From the plot, select the highest learning rate still associated with a falling loss.')
        self.optimizer.load_state_dict(curr_opt_state)
        return

    def plot(self, plot_type='loss'):
        """
        plots training history
        Args:
          plot_type (str):  one of {'loss', 'lr'}
        Return:
          None
        """
        if self.hist is None or len(self.hist.epoch_log) == 0:
            raise Exception('No training history - did you train the model yet?')

        if plot_type == 'loss':
            plt.plot(self.hist.get_epoch_log('loss'), marker='o')
            if self.hist.has_epoch_key('val_loss'):
                plt.plot(self.hist.get_epoch_log('val_loss'), marker='x', markersize=10)
                legend_items = ['train', 'validation']
            else:
                legend_items = ['train']
            plt.title('Model Loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(legend_items, loc='upper left')
            plt.show()
        elif plot_type == 'lr':
            if not self.hist.has_batch_key('lrs'):
                raise ValueError('no lrs in history - did you use an LR scheduler when calling fit?')
            plt.plot(self.hist.get_batch_log('lrs'))
            plt.title('LR Schedule')
            plt.ylabel('lr')
            plt.xlabel('iterations')
            plt.show()
        else:
            raise ValueError('invalid type: choose one of {"loss", "lr"}')
        return


    def save(self, path):
        """
        save model
        """
        torch.save(self.model.state_dict(), path)


    def load(self, path):
        """
        load model
        """
        self.model.load_state_dict(torch.load(path))
        return


    def reset_weights(self):
        """
        randomly reset parameters of model
        """
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'): m.reset_parameters()
        self.model.apply(weight_reset)
        return
