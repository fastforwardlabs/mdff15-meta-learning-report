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
size was 25. At the end of an epoch, we evaluated the model performance on the meta-validation dataset. At the end of 50 epochs, we evaluated the 
model on the meta-test dataset.

![Figure 9: 5-way, 1-shot episode example](figures/9.png)

<<to-do: fix model architecture>>

In terms of model architecture, we used a network with 4 hidden layers—with sizes 28, 207, 20, 20, each including batch normalization and 
ReLU nonlinearities, followed by a linear layer and softmax. For all models, the loss function was the cross-entropy error between the 
predicted and true labels.

The models were trained using an SGD optimizer with a learning rate of 0.001, an inner learning rate of 0.01 for the adaptation process, 
a step size (that is, number of gradient steps) of 5, and a task batch size of 25. All the hyper-parameters were the same for all the models, 
for a consistent comparison.

### Results

<<to-do: update result images, and also update 20 sample runs and corresponding sentences>>

The figures below illustrate how MAML performs on the 100- and 20-item randomly sampled versions of the Quick, Draw! dataset, for a 5-way or a 
10-way classification few-shot problem, with a varying number of examples per class. As expected, the model performance on the blind meta-test set 
is better when the model is trained on a 100-sample subset instead of using just 20 samples. Further, 5-way classification yields better results 
than 10-way classification— which is to be expected, given that 5-way classification is an easier task than 10-way classification. Also, as the 
number of shots/examples per class increase, we see better performance during validation and test time. The validation results for both 5-way and 
10-way 1/5/10-shot learning based on 20 samples show some overfitting issues after some epochs. This could likely be fixed with altering learning 
rates, but we left them as is for easy comparison to the 100-sample dataset results. 

![Figure 10. 5-way, 1/5/10-shot results based on 100 random sampled images](figures/10.png)

![Figure 11. 10-way, 1/5/10-shot results based on 100 random sampled images](figures/11.png)

![Figure 12. 5-way, 1/5/10-shot results based on 20 random sampled images](figures/12.png)

![Figure 13. 10-way, 1/5/10-shot results based on 20 random sampled images](figures/13.png)

Our results demonstrate that the MAML approach is beneficial for learning with only a few examples. In the 100 randomly sampled images scenario, 
the 5-way classification task gives an accuracy of over 60% with just one example on the validation and test datasets. The model performance is 
even better with additional examples; for both 5 examples and 10 examples per class, accuracy shoots up to 80%. As expected, for the 10-way 
classification task, the results are lower (by around 10%) but still promising, which suggests the applicability of the approach for fast 
adaptive learning. <<Todo: add a sentence for 20 random sample scenario>>






