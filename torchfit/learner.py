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

    
    def __init__(self, model, train_loader=None, val_loader=None,
                 criterion=None,        # default is crossentropy
                 optimizer=None,    # default sgd
                 metrics=[accuracy], # default is accuracy
                 seed=None,
                 device=None,
                 use_amp=False,
                 amp_level='O1',
                 gpus=None,
                 verbose=1):
        """
        Learner constructor

        Args:
          model (nn.Module): the network
          train_loader(DataLoader):  training dataset
          val_loader(DataLoader):  validation dataset
          criterion(nn._Loss): loss function. Default is CrossEntropy if None.
          optimizer(Optimizer):  optimizer.  Default is SGD if None.
          metrics(list): list of functions for computing metrics. Default is accuracy.
          seed(int): random seed
          device(str): 'cuda' or 'cpu' or 'cuda:N' where N is integer of GPU
                       If None, inferred automatically.
                       To select a specific GPU, use 'cuda:N' where N is the
                       integer index of the GPU on your system (e.g., 1 is the 
                       second GPU).
          gpus (list of ints):  list of GPUs to use
                                This overrides value supplied for device argument
          use_amp(bool): train using automatic mixed precision.  default:False
          amp_level(str): opt_level for automatic mixed precision.  default:'O1'
                          https://nvidia.github.io/apex/amp.html
          verbose(int): verbosity
        """
        # set seed
        if seed and seed >= 0:
            torch.manual_seed(seed)

        # set device
        if gpus is not None:
            if type(gpus) != list: raise ValueError('gpus must be list of ints')
            if device is not None:
                warnings.warn('device argument is being ignored - using gpus argument')
            if len(gpus) == 1:
                self.gpus = None
                self.device = 'cuda:%s' % (gpus[0])
            else:
                self.gpus = gpus
                self.device = 'cuda'
        else:
            if device is None:
               self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = device
            self.gpus = gpus

        # instance variables
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.verbose = verbose
        self.criterion = criterion
        if self.criterion is None: self.criterion = DEFAULT_LOSS
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

        # multi-gpu
        self.multigpu = False
        if self.gpus is not None and len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)
            self.multigpu = True


    def forward(self, x):
        """
        subclass should override for non-standard forwards
        """
        return self.model(x)


    def compute_loss(self, outputs, targets):
        """
        subclass should override for non-standard scenarios
        """
        return self.criterion(outputs, targets)


    def train_step(self, batch, batch_idx):
        """
        subclass should override for non-standard training steps
        """
        # extract data from batch
        if isinstance(batch[0], (list, tuple)):
            X_batch = [x.to(self.device) for x in batch[0]]
        else:
            X_batch = batch[0].to(self.device) 
        y_batch = batch[1].to(self.device)

        # forward pass
        outputs = self.forward(X_batch)

        # compute loss
        batch_loss = self.compute_loss(outputs, y_batch)
        return {'loss': batch_loss, 'targets':y_batch, 'outputs': outputs}


    def validation_step(self, batch, batch_idx):
        """
        subclass should override for non-standard steps
        """
        return self.train_step(batch, batch_idx)
    

    def test_step(self, batch, batch_idx):
        """
        subclass should override for non-standard setps
        """
        return self.train_step(batch, batch_idx)



    def _fit(self, lr, epochs=1, 
             schedulers=None, accumulation_steps=1, 
             gradient_clip_val=0,
             internal_flag=False):
        """
        train the model
        Args:
          lr (float):  learning rate
          epochs (int):  number of epochs.  default:1
          schedulers(list):  list of LR schedulers.  Default is None.
          accumulation_steps(int): number of batches for gradient accumulation.
                                   default:1
          gradient_clip_val(int): gradient clipping value.  default:0 (no clipping)
          internal_flag(bool): Set to True by methods when invoked
                               by other methods in Learner
        Returns:
          History:  History object containing training history
        """
        self._check_loader()
        opt = self.optimizer

        # history
        self.hist = History()

        # check amp
        if self.use_amp:
            from apex import amp

        # check schedulers
        unk_schedulers=[]
        if schedulers is not None:
            if type(schedulers) != list: raise ValueError('schedulers must be list of _LRScheduler instances')
            for s in schedulers:
                if type(s).__name__ not in SCHED_LOCATIONS:
                    unk_schedulers.append(type(s).__name__)
            #if len(unk_schedulers) > 0:
                #raise ValueError('unknown schedulers  were supplied: %s'  % (' '.join(unk_schedulers)))
        

        # set learning rate        
        if schedulers is None:
            for g in opt.param_groups:
                g['lr'] = lr
        else:
            if not internal_flag:
                warnings.warn('lr parameter will be ignored - using lr setting in ' +\
                              'supplied scheduler and/or optimizer')

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

                dct = self.train_step(batch_data, batch_i)
                y_batch = dct['targets']
                y_batch_pred = dct['outputs']
                batch_loss = dct['loss']

                if self.use_amp:
                    with amp.scale_loss(batch_loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    batch_loss.backward()

                if (batch_i+1) % accumulation_steps == 0:

                    # apply gradient clipping
                    if self.use_amp and gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(opt), gradient_clip_val)
                    elif not self.use_amp and gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)

                    # optimizer step
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
                            if SCHED_LOCATIONS.get(type(s).__name__, None) is None: s.step()
                            elif SCHED_LOCATIONS[type(s).__name__] == 'batchlevel': s.step()

                    # update lr log
                    if schedulers:
                        try:
                            last_lr = schedulers[0].get_last_lr()
                        except:
                           last_lr = schedulers[0].get_lr()
                    else:
                        last_lr = self.optimizer.param_groups[0]['lr']
                    if isinstance(last_lr, (list, tuple, np.ndarray)):
                        last_lr = last_lr[0]
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
                    if SCHED_LOCATIONS[type(s).__name__] == 'epochlevel': s.step()
        return self.hist


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
        return self._fit(lr, epochs=epochs, schedulers=schedulers, accumulation_steps=accumulation_steps)



    def fit_onecycle(self, lr, epochs=1, start_pct=0.25, accumulation_steps=1):
        """
        Train using Leslie Smith's 1cycle policy (https://arxiv.org/pdf/1803.09820.pdf)
        Args:
          max_lr(float):  maximum learning rate (lr for apex of triangle)
          epochs(int): number of epochs to train. Default is 1.
          start_pct(float): percentage of cycle that is increasing.
                            Using <0.5 slants triangle to left as proposed by Howard and 
                            Ruder (2018): https://arxiv.org/abs/1801.06146
                            Default is 0.25.
          accumulation_steps(int): number of batches for gradient accumulation.
                                   default:1
        """
        end_pct = 1-start_pct
        trn = self.train_loader
        scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                base_lr=lr/100, 
                                                max_lr=lr, 
                                                cycle_momentum=False, 
                                                step_size_up=math.floor(epochs*len(trn)*start_pct),
                                                step_size_down=math.floor(epochs*len(trn)*end_pct))
        return self._fit(lr, epochs, schedulers=[scheduler], 
                        accumulation_steps=accumulation_steps, internal_flag=True)



    def predict(self, data_loader, return_targets=False):
        """Generates output predictions for the input samples.

        Args:
          data_loader(DataLoader): DataLoader instance
        Returns:      
          np.ndarray:  NumPy array of predictions
        """
        preds, _, targets = self._predict(data_loader)
        if self.device != 'cpu':
            preds = preds.cpu().detach()
            targets = targets.cpu().detach()
        if return_targets:
            return (preds.numpy(), targets.numpy())
        else:
            return preds.numpy()


    def predict_example(self, data, preproc_fn=None, labels=[]):
        if preproc_fn is not None:
            data = preproc_fn(data)
        if type(data) == list:
            data = [d.to(self.device) for d in data]
        else:
            data = data.to(self.device)
        pred = self.model(data)
        pred = pred.cpu().detach().numpy()
        if labels:
            return labels[np.argmax(pred)]
        else:
            return pred


    def _predict(self, data_loader, test=False):
        """
        Generates output predictions for the input samples
        along with returning loss and ground truth labels

        Args:
            data_loader: DataLoader instance
            test(bool): If True, self.test_step is called
                        instead of self.validation_step.
                        The loss computation might be omitted
                        if self.test_step is overridden, for example.
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

                if test:
                    dct = self.test_step(batch_data, batch_i)
                else:
                    dct = self.validation_step(batch_data, batch_i)
                y_batch = dct['targets']
                y_batch_pred = dct['outputs']
                batch_loss = dct['loss']
                batch_loss = self.criterion(y_batch_pred, y_batch)
                total_loss += batch_loss.item()

                # shapes
                pred_shape = (n,) if len(y_batch_pred.size()) == 1 \
                                  else (n,) + y_batch_pred.size()[1:]
                true_shape = (n,) if len(y_batch.size()) == 1 \
                                  else (n,) + y_batch.size()[1:]

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
        if type(self).__name__ != 'Learner':
            warnings.warn('currently_unsupported: find_lr is not currently supported for '+
                          'subclasses of Learner')
            return
        self._check_loader()
        curr_opt_state = self.optimizer.state_dict()
        self.optimizer.load_state_dict(self.state['optimizer'])
        try:
            import logging
            logging.getLogger('torch_lr_finder').setLevel(logging.CRITICAL)
        except: pass
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #from torch_lr_finder import LRFinder
            from .lr_finder import LRFinder

        opt = self.optimizer

        if use_val_loss and self.val_loader is None:
            raise ValueError('use_val_loss=True but self.val_loader is None')
        val_loader = None
        if use_val_loss:
            val_loader = self.val_loader
        lr_finder = LRFinder(self.model, opt, self.criterion, device=self.device)

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
        self.model.to(self.device)
        if self.multigpu: 
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)

        return


    def reset_weights(self):
        """
        randomly reset parameters of model
        """
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'): m.reset_parameters()
        self.model.apply(weight_reset)
        return


    def _check_loader(self, use_val=False):
        if use_val:
            loader_name = 'val_loader'
            loader = self.val_loader
        else:
            loader_name = 'train_loader'
            loader = self.train_loader

        if loader is None:
            raise ValueError('%s is required for this method' % (loader_name))
