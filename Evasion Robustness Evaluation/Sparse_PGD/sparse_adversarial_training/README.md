# Sparse-PGD: An Effective and Efficient Attack for $l_0$ Bounded Adversarial Perturbation

## Requirements

To execute the code, please make sure that the following packages are installed:

- [NumPy](https://docs.scipy.org/doc/numpy-1.15.1/user/install.html)
- [PyTorch and Torchvision](https://pytorch.org/) (install with CUDA if available)
- [matplotlib](https://matplotlib.org/users/installing.html)
- [robustbench]()

## Executing the code

### Advversarial Training

Run the following command to train PreActResNet18 with sAT or sTRADES on CIFAR10:

```
python train.py --exp_name debug --data_name cifar10 --data_dir [DATA PATH] --model_name preactresnet --max_epoch 40 --lr 0.05 --eval_samples 100 --train_loss [adv, trades] -k 20 --n_iters 100 --gpu 0
```

- exp_name: experiment name
- data_name: cifar10 or cifar100 (we only support cifar10 and cifar100 now)
- data_dir: path to the dataset
- model_name: choose a model for training
- max_epoch: number of epochs for training
- lr: initial learning rate
- eval_samples: number of examples to evaluate
- train_loss: choose a loss for training from 'adv' (sAT) and 'trades' (sTRADES)
- k: $l_0$ norm budget
- n_iters: number of iterations for sPGD
- gpu: gpu id

## Acknowledgement

Parts of codes are based on [DengpanFu/RobustAdversarialNetwork: A pytorch re-implementation for paper "Towards Deep Learning Models Resistant to Adversarial Attacks" (github.com)](https://github.com/DengpanFu/RobustAdversarialNetwork)
