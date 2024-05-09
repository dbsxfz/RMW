# Code for Watermark Embedding

This folder contains the implementation for watermark embedding, conceptualized as learning the watermarked model as a "friendly teacher":

- The Jupyter notebook `cifar10_base.ipynb` details the training process for a base model on CIFAR-10. During our watermark embedding process, we initially use the standard training process for 70 epochs to produce a model that serves as a surrogate for generating the UAE trigger set and control group. Subsequently, we fine-tune the model for an additional 30 epochs to embed the watermark.

- The Jupyter notebook `embedding.ipynb` illustrates the UAE watermark embedding process, which involves learning the watermarked model as a "friendly teacher". This includes optimizing function mapping properties and employing sharpness-aware minimization techniques.

- The Jupyter notebook `extraction.ipynb` demonstrates an example of model extraction. The final watermark accuracy is calculated using the formula `wm1_acc - wm2_acc`. This notebook showcases the adjustment of the watermarked model's output distribution using temperature scaling. Originally, this was implemented by inserting a layer at the end of the model that divides the logits; for convenience, we implement this equivalent form directly in the extraction function as a conceptual demonstration.