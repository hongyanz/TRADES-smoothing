# TRADES-random-smoothing

This is the code for the paper "Random Smoothing Might be Unable to Certify ![](http://latex.codecogs.com/gif.latex?\ell_\infty) Robustness for High-Dimensional Images".

## Prerequisites
* Python (3.6.4)
* Pytorch (0.4.1)
* CUDA
* numpy

## Install
We suggest to install the dependencies using Anaconda or Miniconda. Here is an exemplary command:
```
$ wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
$ bash Anaconda3-5.1.0-Linux-x86_64.sh
$ source ~/.bashrc
$ conda install pytorch=0.4.1
```
## TRADES+random smooth: A New Training Method for Certifiable Robustness

We used TRADES to train a soft-random-smoothing classifier by injecting Gaussian noise. The method minimizes empirical risk of a regularized surrogate loss L(.,.) (e.g., the cross-entropy loss) with Gaussian noise:
![](http://latex.codecogs.com/gif.latex?\min_{f}\mathbb{E}_{X,Y}\mathbb{E}_{\eta\sim\mathcal{N}(0,\sigma^2I)}\left[\mathcal{L}(f(X+\eta),Y)+\beta\max_{X'\in\mathbb{B}_2(X,\epsilon)}\mathcal{L}(f(X+\eta),f(X'+\eta))\right].)

## Running Demos

### Certified ![](http://latex.codecogs.com/gif.latex?\ell_\infty) robustness:

* Train ResNet-110 model on CIFAR10:
```bash
  $ python train_trades_dim.py
```

### Effectiveness of lower bound:

* Train ResNet-110 model on CIFAR10:
```bash
  $ python train_trades_dim.py
```

<p align="center">
    <img src="results/vary_dim_cifar10_trades.png" width="450"\>
</p>
<p align="center">
