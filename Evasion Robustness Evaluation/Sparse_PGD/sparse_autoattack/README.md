# Sparse-PGD: An Effective and Efficient Attack for $l_0$ Bounded Adversarial Perturbation

## Requirements

To execute the code, please make sure that the following packages are installed:

- [NumPy](https://docs.scipy.org/doc/numpy-1.15.1/user/install.html)
- [PyTorch and Torchvision](https://pytorch.org/) (install with CUDA if available)
- [matplotlib](https://matplotlib.org/users/installing.html)
- [robustbench]()

## Executing the code

### Sparse-AutoAttack

Run the following command to evaluate sparse-AutoAttack on CIFAR10:

```
python evaluate.py --dataset cifar10 --data_dir [DATASET PATH] --model [standard, l1, linf, l2, l0] --ckpt [CHECKPOINT NAME OR PATH] -k 20 --n_iters 300 --n_examples 10000 \
                   --gpu 0 --bs 512 [--unprojected_gradient]
```

- dataset: cifar10 or cifar100 (we only support cifar10 and cifar100 now)
- data_dir: path to the dataset
- model: choose a model from standard, l1, linf, l2, l0
- ckpt: checkpoint name (for robustbench models) or checkpoint path (for vanilla, $l_1$ and $l_0$ models)
- k: $l_0$ norm budget
- n_iters: number of iterations
- n_examples: number of examples to evaluate
- unprojected_gradient: optional, whether to use unprojected gradient in the first white-box attack

## Acknowledgement

Codes of models are based on [DengpanFu/RobustAdversarialNetwork: A pytorch re-implementation for paper "Towards Deep Learning Models Resistant to Adversarial Attacks" (github.com)](https://github.com/DengpanFu/RobustAdversarialNetwork) and [IVRL/FastAdvL1 (github.com)](https://github.com/IVRL/FastAdvL1)

Codes of Sparse-RS are from [fra31/sparse-rs: Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks (github.com)](https://github.com/fra31/sparse-rs)
