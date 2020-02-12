# TorchFit

`TorchFit` is a bare-bones, minimalistic *training-helper* for **PyTorch** that exposes an easy-to-use `fit` method in the style of **fastai** and **Keras**.  

`TorchFit` is intended to be minimally-invasive with a tiny footprint and as little bloat as possible. It is well-suited to those that are new to training models in PyTorch. 

## Usage

```python


# normal PyTorch stuff
train_loader = create_your_training_data_loader()
val_loader = create_your_validation_data_loader()
test_loader = create_your_test_data_loader()
model = create_your_pytorch_model()

# wrap model and data in Learner
import torchfit
learner = torchfit.Learner(model, train_loader, val_loader=val_loader)

# estimate LR using Learning Rate Finder
learner.find_lr()

# train using 1cycle learning rate policy
learner.fit_onecycle(1e-4, 3)

# plot training vs. validation loss
learner.plot('loss')

# make predictions as easy as in Keras
y_pred = learner.predict(test_loader)

# save model and reload later
learner.save('/tmp/mymodel')
learer.load('/tmp/mymodel')
```


#### `TorchFit` Training Loop
<img src="https://github.com/amaiya/torchfit/raw/develop/images/torchfit_progress.gif" width="800">


## Tutorials and Examples
- **[Quickstart with MNIST](https://github.com/amaiya/torchfit/blob/master/examples/quickstart-mnist.ipynb):**  quickstart notebook to get you up and running
- **[Tutorial Notebook](https://github.com/amaiya/torchfit/blob/master/examples/tutorial.ipynb):**  tutorial notebook using the same model and data employed in the [PyTorch text classification tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)


##  Features

#### Learning Rate Finder
```learner.find_lr()```


#### A `fit` method for Training
```
# Examples
learner.fit(lr, epochs)
learner.fit_onecycle(lr, epochs)
learner.fit(lr, epochs, schedulers=[scheduler])
```

#### Easy-to-Execute Testing and Predictions
```
# Examples
outputs = learner.predict(test_loader)
outputs, targets = learner.predict(test_loader, return_targets=True)

text = 'Shares of IBM rose today.'
predicted_label = learner.predict_example(text, preproc_fn=preprocess, labels=labels)
```


#### Gradient Accumulation
```learner.fit_onecycle(lr, 1, accumulation_steps=8)```


#### Gradient Clipping
```learner.fit_onecycle(lr, 1, gradient_clip_val=1)```


#### Mixed Precision Training
```torchfit.Learner(model, train_loader, val_loader=val_loader, use_amp=True, amp_level='O2')```

#### Multi-GPU Training and GPU Selection

To train on first two GPUs (0 and 1):

```learner = torchfit.Learner(model, train_loader, val_loader=test_loader, gpus=[0,1])```

To train only on the second GPU, one can do either this:

```learner = torchfit.Learner(model, train_loader, val_loader=test_loader, gpus=[1])```

or this...

```learner = torchfit.Learner(model, train_loader, val_loader=test_loader, device='cuda:1')```


#### Resetting Weights of Model
```learner.reset_weights()``` 


#### Saving/Loading Model
```
learner.save('/tmp/mymodel')
learner.load('/tmp/mymodel')
```






## Installation

After ensuring [PyTorch is installed](https://pytorch.org/get-started/locally/), install `TorchFit` with:

```
pip3 install torchfit

```

<!-- pip3 install pillow==6.2.2 torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html -->
