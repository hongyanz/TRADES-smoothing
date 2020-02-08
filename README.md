# TRADES-random-smoothing

This is the code for the paper "Random Smoothing Might be Unable to Certify ![](http://latex.codecogs.com/gif.latex?\ell_\infty) Robustness for High-Dimensional Images" by Avrim Blum (TTIC), Travis Dick (U. of Pennsylvania), Naren Manoj (TTIC), and Hongyang Zhang (TTIC) (ordered alphabetically).

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
## TRADES+random smooth: A New Training Method with Certifiable Robustness

### What is TRADES + random smoothing?
We used TRADES to train a soft-random-smoothing classifier by injecting Gaussian noise. The method minimizes:
![](http://latex.codecogs.com/gif.latex?\min_{f}\mathbb{E}_{X,Y}\mathbb{E}_{\eta\sim\mathcal{N}(0,\sigma^2I)}[\mathcal{L}(f(X+\eta),Y)+\beta\max_{X'\in\mathbb{B}_2(X,\epsilon)} \mathcal{L}(f(X+\eta),f(X'+\eta))])
