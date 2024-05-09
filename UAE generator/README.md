# Code for Generating UAEs as the Trigger Set

This folder contains the implementation for generating Universal Adversarial Examples (UAEs):

- The implementation leverages a diffusion backbone model, adapted from [this repository](https://github.com/FutureXiang/Diffusion).

- The Jupyter notebook `SDEdit.ipynb` demonstrates the application of this code for synthesizing UAEs targeting the CIFAR-10 dataset using the Adversarial Edition technique, as illustrated in Section 4.1.2 of the associated paper. To reduce generation time, this conceptual version omits the use of Adversarial Warm-up, Adversarial Guidance (implementation can be found in the script `./model/EDM_guide.py`), and UAE Selection.
