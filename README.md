# TorchFit

`TorchFit` is a bare-bones, minimalistic *training-helper* for **PyTorch** that exposes an easy-to-use `fit` method in the style of **fastai** and **Keras**.  It is intended to be easy-to-use with a tiny footprint and as little bloat as possible. `TorchFit` is particularly well-suited to those new to PyTorch trying to train models. For more complex training scnenarios (e.g., training GANs, multi-node GPU training), [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is highly recommended.


## Usage

```python


# normal PyTorch stuff
train_loader = load_your_training_data_loader()
val_loader = load_your_validation_data_loader()
model = create_your_pytorch_model()
def accuracy(y_true, y_pred):
    return np.mean(y_true.numpy() == np.argmax(y_pred.numpy(), axis=1))

# wrap model and data in torchfit.Learner
import torchfit
learner = torchfit.Learner(model, train_loader, val_loader=test_loader)

# estimate LR using fastai-like Learning Rate Finder
learner.find_lr()

# train using learning rate
learner.fit_onecycle(1e-4, 3)

# plot training vs. validaiton loss
learner.plot('loss')

# make predictions
learner.predict(val_data_loader)

# save model and reload lader
learner.save('/tmp/mymodel')
learer.load('/tmp/mymodel')
```

For more information see: `tutorial.ipynb`


## Usage

```
pip3 install torchfit

```
