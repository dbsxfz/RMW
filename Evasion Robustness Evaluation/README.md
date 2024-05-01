# Code for Evaluating Evasion Robustness of Watermarked Models

This folder provides the implementation for assessing the evasion robustness of watermarked models:

- `pixel_backdoor.py`: Implements the techniques from "Better Trigger Inversion Optimization in Backdoor Scanning." (Pixel Backdoor)
- `Sparse_PGD/`: Contains the official implementation of "Sparse-PGD: An Effective and Efficient Attack for $l_0$ Bounded Adversarial Perturbation."

We have made enhancements to the vanilla Pixel Backdoor, including the addition of a universal attack version that flips samples from all classes to a single target class and an untargeted attack version that flips all samples to arbitrary incorrect labels. Furthermore, control over the intensity of regularization terms has been introduced.

The notebooks `evaluation_normal_imagenette.ipynb` and `evaluation_pair_imagenette.ipynb` provide example usage of the code for untargeted attacks and fine-grained evaluation from source to target classes, respectively. These correspond to Section 5.1.1 "Overall Evaluation" and Section 5.1.2 "Class-wise Evaluation of Backdoor Watermarking" in the main text. Hyperparameters for initializing attacks are also provided.
