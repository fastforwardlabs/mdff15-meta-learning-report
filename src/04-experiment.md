## Experiment

[The MAML paper](https://arxiv.org/pdf/1703.03400.pdf) explores the approach for multiple problems: regression, classification, and reinforcement
learning. To gain a better understanding of the algorithm and investigate whether MAML really learns to adapt to novel tasks, we tested the 
technique on the [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset. All of the experiments were performed using PyTorch (which allows 
for automatic differentiation of the gradient updates), along with the [torch-meta](https://github.com/tristandeleu/pytorch-meta) library. The 
torch-meta library provides data loaders for few-shot learning, and extends PyTorch’s Module class to simplify the inclusion of additional 
parameters for different modules for meta-learning. This functionality allows one to backpropagate through an update of parameters, which is a key 
ingredient for gradient-based meta-learning. While torch-meta provides an excellent structure for creating reproducible benchmarks, it will be 
interesting to see its integration with other meta-learning approaches that handle datasets differently, and its flexibility in adopting them in 
the future. For our purposes, we extended the torch-meta code to accommodate the Quick, Draw! data set-up. The experiment code is available 
[here]().

### Dataset

The Quick, Draw! dataset consists of 50 million doodles (hand-drawn figures) across 345 categories. We conducted two experiments: in one, we 
randomly selected 100 images; in the other, we randomly selected 20 images per class. The 345 classes were randomly split into 
meta-train/validation/test datasets as 207/69/69. The training and evaluation was performed on the meta-training set. The meta-validation set was 
mostly used for hyper-parameter tuning, and the meta-test set measured the generalization to new tasks.

### Set-up

We evaluated the MAML approach on 5-way 1/5/10-shot and 10-way 1/5/10-shot settings for the Quick, Draw! dataset. An experiment on each of the 
100-sample and 20-sample datasets consisted of training for 50 epochs with each epoch consisting of 100 batches of tasks, where a task’s batch 
size was 25 for 100-sample and 10 for 20-sample datasets. At the end of an epoch, we evaluated the model performance on the meta-validation dataset. At the end of 50 epochs, we evaluated the 
model on the meta-test dataset.

![Figure 9: 5-way, 1-shot episode example](figures/9.png)

In terms of model architecture, we used a network with 4 convolution layers—with size 20 channels in the intermediate representations, each 
including batch normalization and ReLU nonlinearities, followed by a linear layer. For all models, the loss function was the cross-entropy error between the predicted and true labels.

The models for the 100-sample dataset were trained using an SGD optimizer with a learning rate of 0.001, an inner learning rate of 0.01 for the 
adaptation process, a step size (that is, number of gradient steps) of 5, and a task batch size of 25. All the hyper-parameters were the same for 
all the models, for a consistent comparison. While the models for the 20-sample dataset were trained with a slightly lower learning rate of 0.0005, an inner learning rate of 0.005,  a task batch size of 10 along with the rest of the parameters were same as the 100-sample dataset.

### Results

The figures below illustrate how MAML performs on the 100- and 20-item randomly sampled versions of the Quick, Draw! dataset, for a 5-way or a 
10-way classification few-shot problem, with a varying number of examples per class. As expected, the model performance on the both the 
meta-validation and meta-test set is better when the model is trained on a 100-sample subset instead of using just 20 samples. Further, 
5-way classification yields better results 
than 10-way classification— which is to be expected, given that 5-way classification is an easier task than 10-way classification. Also, as the 
number of shots/examples per class increase, we see better performance during validation and test time. The validation results for 5-way 
1/5/10-shot learning based on 20 samples look promising too. In the 10-way learning based on 20-samples we see some overfitting after a few epochs and may want to restrain the model by stopping early. That said, we have left them as is for easy comparison with the rest of the experiment results.

![Figure 10. 5-way, 1/5/10-shot results based on 100 random sampled images](figures/10.png)

![Figure 11. 10-way, 1/5/10-shot results based on 100 random sampled images](figures/11.png)

![Figure 12. 5-way, 1/5/10-shot results based on 20 random sampled images](figures/12.png)

![Figure 13. 10-way, 1/5/10-shot results based on 20 random sampled images](figures/13.png)

![Figure 14. Meta-test dataset results](figures/14.png)

Our results demonstrate that the MAML approach is beneficial for learning with only a few examples. In the 100 randomly sampled images scenario, 
the 5-way classification task gives an accuracy of around 68% with just one example. The model performance is 
even better with additional examples; for both 5 examples and 10 examples per class, accuracy shoots over 80%. As expected, for the 10-way 
classification task, the results are lower (by around 10-15%) but still promising. 

For the 20-random sample scenario and a more realistic one from a meta-learning point of view, the 5-way results are still pretty good ~60% accuracy with just one example. The 10-way classification results are lower similar to the 100-sample dataset. Nonetheless, overall the results are promising even with minimal tuning and suggests the applicability of the approach for fast adaptive learning.
