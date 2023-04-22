# DECONET
Code for the experiments accompanying the publication "DECONET: an Unfolding Network for Analysis-based Compressed Sensing with Generalization Error Bounds", to appear in IEEE Transactions on Signal Processing.

The repo consists of the following 4 .py files: 
a) `deconet_mnist.py` builds the DECONET model, which is then trained and tested on the MNIST dataset
b) `deconet_cifar.py` builds the DECONET model, which is then trained and tested on the CIFAR10 dataset
c) `synthetic_dataset.py` creates synthetic data drawn from the normal distribution
d) `deconet_synthetic.py` builds the DECONET model, which is then trained and tested on the synthetic dataset created in (c)


## Train model

In order to run each script, the user must supply a set of arguments, with default values listed in an argparse environment, in each of the three deconet scripts. For example, if one wishes to train and test DECONET on CIFAR10, with 20 layers, redundancy 30 * 1024, kaiming initialization and 50% CS ratio, the desired command will be the following:


```
python deconet_cifar.py --layers 20 --red 30720 --meas 0.5 --init kaiming 
```
