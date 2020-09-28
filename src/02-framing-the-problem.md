## Framing the problem

What kind of problems can meta-learning help us solve? One of the most popular categories is few-shot learning. In a few-shot learning scenario, we have only a limited number of examples on which to perform supervised learning, and it is important to learn effectively from them. The ability to do so could help relieve the data-gathering burden (which at times may not even be possible).

Let’s say we want to solve a few-shot classification problem, shown below in Figure 4. Usually the few-shot classification problem is set up as a N-way k-shot problem, where N is the number of classes and k is the number of examples in each class. For example, let's say we are given an image from each of five different classes (that is, N=5 and k=1) and we are supposed to classify new images as belonging to one of these classes. What can we do? How would one normally model this?

![Figure 4: A few-shot classification (5-way, 1-shot) problem](figures/ff15-47.png)

One way to solve the problem would be to train a neural network model from scratch on the five training images. At a high level, a training step 
will look something like Figure 5, below. The neural network model is randomly initialized and receives an image (or images) as input. 
It then predicts the output label(s) based on the initial model parameters. The difference between the true label(s) and the predicted label(s) 
is measured by a loss function (for example, cross-entropy), which in turn is used to compute the gradients. The gradients are then used to help 
calculate new model parameters that best reduce the difference between the “true” and predicted labels. This entire step is known as 
backpropagation. After backpropagation, the optimizer updates the model parameters for the model, and all of these steps are repeated for the 
rest of the images and/or for some number of epochs, until the loss, evaluated on the train or test data, falls below an acceptable level. 

![Figure 5: A training step in normal training process - adopted from HuggingFace’s blog post, [“From zero to research”](https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a#0f06)](figures/ff15-48.png)

With only five images available for training, chances are that the model would likely overfit and perform poorly on the test images. Adding some 
regularization or data augmentation may alleviate this problem to some extent, but it will not necessarily solve it. The very nature of a 
few-shot problem makes it hard to solve, as there is no prior knowledge of the tasks. 

Another possible way to solve the problem could be to use a pre-trained network from another task, and then fine-tune it on the five training 
images. However, depending on the problem, this may not always be feasible, especially if the task the network was trained on differs substantially.
