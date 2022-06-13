# Code explanation

This repository contains information, code from the manuscript [Deep-Learning-Based Multipath Time of Arrival
Estimation using Sparse Kernel Representation], which is based on Pytorch. And the code contains the core content of the proposed algorithm.

## Sparse coding representation

data.re.py lists ont-hot, correlation and gaussian coding for off-grid time delay representation.

### Model of network
The network of proposed method as shown in modules.py.


### Parameters of simulation signal as shown in data.data.py

 AS shown in data.data.py, we generate the simulation chirp signal with parameters as:

### Train

[`train.py`](train.py) provides the code for training a model from scratch. 




