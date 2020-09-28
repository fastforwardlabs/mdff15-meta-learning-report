## Solving the problem

### Data set-up

What meta-learning proposes is to use an end-to-end deep learning algorithm that can learn a representation better suited for few-shot learning. 
It is similar to the pre-trained network approach, except that it learns an initialization that serves as a good starting point for the handful of 
training data points. In the few-shot classification problem  discussed, we could leverage training data that’s available from other image 
classes; for instance, we could look at the training data available and use images from classes like mushrooms, dogs, eyewear, etc. The model 
could then build up prior knowledge such that, at inference time, it can quickly acquire task-specific knowledge with only a handful of training 
examples. This way, the model first learns parameters from a training dataset that consists of images from other classes, and then uses those 
parameters as prior knowledge to tune them further, based on the limited training set (in this case, the one with five training examples). 

Now the question is, how can the model learn a good initial set of parameters which can then be easily adapted to the downstream tasks? 
The answer lies in a simple training principle, which was initially proposed by Vinyals et. al.^[[Matching Networks for One-Shot Learning](https://arxiv.org/abs/1606.04080)]:

::: info

<center>Train and test conditions must match</center>

::: 

The idea is to train a model by showing it only a few examples per class, and then test it against examples from the same classes that have been 
held out from the original dataset, much the way it will be tested when presented with only a few training examples from novel classes. Each 
training example, in this case, is comprised of pairs of train and test data points, called an *episode*.

![Figure 6: Meta-learning data set-up, adopted from [Optimization as a Model for Few-Shot Learning (PDF)](https://openreview.net/pdf?id=rJY0-Kcll)](figures/ff15-49.png)

This is a departure from the way that data is set up for conventional supervised learning. The training data (also called the meta-training data) 
is composed of train and test examples, alternately referred to as the support and query set.

::: info

The number of classes *(N)* in the support set defines a task as an *N*-class classification task or *N*-way task, and the number of labeled 
examples in each class *(k)* corresponds to *k*-shot, making it an *N*-way, *k*-shot learning problem.

:::

In this case, we have a 5-way, 1-shot learning problem. 

Similar to conventional supervised learning, which sets aside validation and test datasets for hyper-parameter tuning and generalization, 
meta-learning also has meta-validation and meta-test sets. These are organized in a similar fashion as the meta-training dataset in episodes, 
each with support and query sets; the only difference is that the class categories are split into meta-training, validation, and test datasets, 
such that the classes do not overlap.

### Meta-learning: learning to learn

A meta-learning model should be trained on a variety of tasks, and then optimized further for novel tasks. A task, in this case, is basically a 
supervised learning problem (like image classification or regression). The idea is to extract prior information from a set of tasks that allows 
efficient learning on new tasks. For our image classification problem, the ideal set-up would include many classes, with at least a few examples 
for each. These can then be used as a meta-training set to extract prior information, such that when a new task (like the one in the Figure 4, 
above) comes in, the model can perform it more efficiently.

At a high level, the meta-learning process has two phases: meta-learning and adaptation. In the meta-learning phase, the model learns an initial 
set of parameters slowly across tasks; during the adaptation phase, it focuses on quick acquisition of knowledge to learn task-specific 
parameters. Since the learning happens at two levels, meta-learning is also known as learning to learn.^[Thrun S., Pratt L. (eds). [Learning to Learn](https://link.springer.com/chapter/10.1007/978-1-4615-5529-2_1). Springer, Boston, MA. 1998.] 

A variety of approaches have been proposed that vary based on how the adaptation portion of the training process performs. These can broadly be classified into three categories: “black-box” or model-based, metric-based, and optimization-based approaches. 

“Black-box” (or model-based) approaches simply train an entire neural network, given some training examples in the support set and an initial 
set of meta-parameters, and then make predictions on the query set. They approach the problem as supervised learning, although there are 
approaches that try to eliminate the need to learn an entire network.^[[One-shot Learning with Memory-Augmented Neural Networks](https://link.springer.com/chapter/10.1007/978-1-4615-5529-2_1)]

Metric-based approaches usually employ non-parametric techniques (for example, *k*-nearest neighbors) for learning. The core idea is to learn a 
feature representation (e.g., learning an embedding network that transforms raw inputs into a representation which allows similarity comparison 
between the support set and the query set). Thus, performance depends on the chosen similarity metric (like cosine similarity or euclidean 
distance).

Finally, optimization-based approaches treat the adaptation part of the process as an optimization problem. This report mainly focuses on one of 
the well-known approaches in this category, but before we delve into it, let’s look at how optimization-based learning actually works.

During training, we iterate over datasets of episodes. In meta-training, we start with the first episode, and the meta-learner takes the training 
(support) set and produces a learner (or a model) that will take as input the test (query) set and make predictions on it. The meta-learning 
objective is based on a loss (for example, cross-entropy) that is derived from the test or query set examples and will backpropagate through these 
errors. The parameters of the meta-learner (that is, meta-parameters) are then updated based on these errors to optimize the loss.^[Note that this 
differs from a conventional supervised learning set-up, in which the objective is based on a loss derived only from the training set, and, of 
course, there is no support or query set!]

In the next step, we look at the next episode, train on the support set examples, make predictions on the query set, update meta-parameters, and 
repeat. In attempting to learn a meta-learner this way, we are trying to solve the problem of generalization. The examples in the test (or query) 
set are not part of the training—so, in a way, the meta-learner is learning to extrapolate.

![Figure 7: Learning to learn](figures/ff15-50.png)

### Model Agnostic Meta-learning (MAML)

Now that we have a general idea of how meta-learning works, the rest of this report mainly focuses on MAML^[[Model Agnostic Meta-learning for Fast Adaptation of Deep Networks (PDF)](https://arxiv.org/pdf/1703.03400.pdf)], which is perhaps one of the best known optimization-based approaches. 
While there have been more recent extensions to it, MAML continues to serve as a foundational approach. 

The goal of meta-learning is to help the model quickly adapt to or learn on a new task, based on only a few examples. In order for that to happen, 
the meta-learning process has to help the model learn from a large number of tasks.  For example, for the image classification problem we’ve 
considered, the new task is the one shown in Figure 4, while the large number of tasks could be images from other classes that are utilized for 
building a meta-training dataset, as shown in Figure 6.

The key idea in MAML is to establish initial model parameters in the meta-training phase that maximize its performance on the new task. This is 
done by updating the initial model parameters with a few gradient steps on the new task. Training the model parameters in this way allows the 
model to learn an internal feature representation that is broadly suitable for many tasks—the intuition being that learning an initialization that 
is good enough, and then fine-tuning the model slightly, will produce good results.

Imagine we have two neural network models that share the same model architecture:^[While it is possible to have two duplicate models that can 
share parameter tensors in popular deep learning frameworks like PyTorch, libraries like [torch-meta](https://github.com/tristandeleu/pytorch-meta)
 have extended the existing torch modules to allow storing additional/new parameters.] *learner* for the meta-learning process and *adapter* for 
the adaptation process. Since we have two models to train, we also have two different learning rates associated with them. The MAML algorithm can 
then be summarized in the following steps:

::: info

- Step 1: Randomly initialize the learner
- Step 2: Repeat the entire process from Step 2.a to Step 3 for all the episodes of the meta-training dataset (or for a certain number of epochs) until the learner converges to a good set of “meta-parameters”
	- Step 2.a: Sample a batch of episodes from the meta-training dataset
	- Step 2.b: Initialize the adapter with the learner’s parameters
	- Step 2.c: While number of inner training steps is not equal to zero
		- Step 2.c.1: Train the adapter based on the support set(s) of the batch, compute the loss and the gradients, and update the adapter’s parameters
	- Step 2.d: Use the updated parameters of the adapter to compute the “meta-loss” based on the query set(s) of the batch
- Step 3: Compute the “meta-gradients”, followed by the “meta-parameters” based on the “meta-loss,” and update the learner’s parameters

:::

The “meta-loss” indicates how well the model is performing on the task. In effect, the *learner* is being fine-tuned using a gradient-based 
approach for every new task in the batch of episodes. Further, the *learner* acts as initialization parameters for the *adapter* so that it can 
perform task-specific learning.

![Figure 8: MAML](figures/ff15-51.png)

During inference, we actually use the meta-trained model (*learner*) to predict on the meta-test set, except this time—although the meta-trained 
model undergoes additional gradient steps to help classify the query set examples—the *learner* parameters aren’t updated.

As long as the model is trained using gradient descent, the approach does not place any constraints on the model architecture or the loss 
function. This characteristic makes it applicable to a wide variety of problems, including regression, classification, and reinforcement learning. 
Further, since the approach actually undergoes a few gradient steps for a novel task, it allows the model to perform better on out-of-sample data, 
and hence achieves better generalization. This behavior can be attributed to the central assumption of meta-learning: that the tasks are 
inherently related and thus data-driven inductive bias can be leveraged to achieve better generalization.
 
