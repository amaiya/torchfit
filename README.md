# TorchFit

`TorchFit` is a bare-bones, minimalistic *training-helper* for **PyTorch** that exposes an easy-to-use `fit` method in the style of **fastai** and **Keras**.  

`TorchFit` is intended to be easy-to-use with a tiny footprint and as little bloat as possible. It is particularly well-suited to those that are new to training models in PyTorch. For more complex training scenarios (e.g., training GANs, multi-node GPU training), [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is highly recommended.


## Usage

```python


# normal PyTorch stuff
train_loader = create_your_training_data_loader()
val_loader = create_your_validation_data_loader()
test_loader = create_your_test_data_loader()
model = create_your_pytorch_model()

# wrap model and data in torchfit.Learner
import torchfit
learner = torchfit.Learner(model, train_loader, val_loader=val_loader)

# estimate LR using fastai-like Learning Rate Finder
learner.find_lr()

# train using learning rate
learner.fit_onecycle(1e-4, 3)

# plot training vs. validaiton loss
learner.plot('loss')

# make predictions
y_pred = learner.predict(test_loader)

# save model and reload lader
learner.save('/tmp/mymodel')
learer.load('/tmp/mymodel')
```

For more information see: `tutorial.ipynb`


## Usage

```
pip3 install torchfit

```
