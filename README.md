
## Introduction

This repository contains the implementaion for the paper:

**[Knowledge Distillation Inspired Fine-Tuning Of Tucker Decomposed CNNS and Adversarial Robustness Analysis](https://ieeexplore.ieee.org/document/9190672)**
Ranajoy Sadhukhan, Avinab Saha, Dr. Jayanta Mukhopadhyay, Dr. Amit Patra
_IEEE International Conference on Image Processing (ICIP) 2020_

## Data preparation

   * Download train-val sets of Image-ILSVRC12 for experiments on ImageNet
   * Update data path in run script
   ```
    conda env update -n tensordecomp -f environment.yml
    source activate tensordecomp
    pip install -e .
```
   * Put cifar10 and cifar100 files cifar-10-python.tar.gz and cifar-100-python.tar.gz within same folder cifar


## Installation

The training/testing environment can be initialized using conda as:
```
conda env update -n tensordecomp -f environment.yml
source activate tensordecomp
pip install -e .
```

## Tensor Decomposition methods

There are two Tensor Decomposition methods implemented here

* CP Decomposition
* Tucker Decomposition

## Adversarial Attacks

The original paper addresses [DeepFool attack](https://arxiv.org/abs/1511.04599) only.
This repository extends experiments on two more adversarial attacks
   * [PGD attack](https://arxiv.org/abs/1706.06083)
   * [Carlini-Wagner attack](https://arxiv.org/abs/1608.04644)

We have used [Foolbox](https://foolbox.readthedocs.io/en/v2.4.0/) for the implementations of these attacks

## Running the code

Update the model and dataset information in ``TensorDecomp/config/default.py`` accordingly
```
cd TensorDecomp
chmod +x run.sh
./run.sh
```

For decomposing the network, use a pretrained undecomposed checkpoint
Update ``run.sh`` script as follows
```
python main.py --pretrained --decompose --gpu <device_id>
```
Update architecture name and the type of loss function in the config file `` TensorDecomp/config/default.py``
In order to use logits loss or KL divergence loss for implementing Knowledge Distillation, 
Update ``run.sh`` script as follows
```
python main.py --pretrained --decompose --teacher --gpu <device_id>
```
Update ``_C.SOLVER.LOSS`` as 'L2' or 'L1' or 'KD' in `` TensorDecomp/config/default.py`` to implement appropriate loss function.

