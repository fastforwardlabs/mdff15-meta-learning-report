## Challenges and ways to overcome

The MAML approach fine-tunes its model using gradient descent each time for a new task. This requires it to backpropagate the meta-loss through 
the model’s gradients, which involves computing derivatives of derivatives, i.e., second derivatives. While the gradient descent at test time helps it extrapolate better, it does have its costs.

Backpropagating through many inner steps can be compute and memory intensive. With only a few gradient steps, it might be a less time-consuming 
endeavor, but it may not be the best solution for scenarios that require a higher number of gradient steps at test time. That said, the authors of 
the [MAML paper](https://arxiv.org/pdf/1703.03400.pdf) also propose a first-order approximation that eliminates the need to compute the second 
derivatives, with a comparable performance. Another closely related work is OpenAI’s Reptile;^[[On First-Order Meta-Learning Algorithms](https://arxiv.org/pdf/1803.02999.pdf)] it builds on first-order MAML, but doesn’t need to split the episode into support and query sets, making it a 
natural choice in certain settings. However, experiments suggest that approaches to reduce computation time while not sacrificing generalization 
performance are still in the works.^[[How to train your MAML](https://arxiv.org/pdf/1810.09502.pdf)]

As we saw previously, learning occurs in two stages: gradual learning is performed across tasks, and rapid learning is performed within tasks. 
This requires two learning rates, which introduces difficulty in choosing hyper-parameters that would help achieve training stability. 
The two learning rates introduce hyper-parameter grid search computation, and hence, time and resources. It is also important to select the learning rate for the adaptation process carefully because it is learning over only a few examples. In that regard, some solutions or extensions to MAML 
have been developed to reduce the need for grid search or hyper-parameter tuning. For example, Alpha MAML^[[Alpha MAML: Adaptive Model Agnostic Meta Learning](https://arxiv.org/abs/1905.07435)] eliminates the need to tune both the learning rates by automatically updating them as needed. MAML++, on the other hand, proposes updating the query set loss (meta-loss) for every training step in the adaptation process, which can help get rid of the training instabilities. In addition, they suggest various other steps to make it computationally efficient.

Research that performs neural architecture search for gradient-based meta-learners^[[Auto-Meta: Automated Gradient Based Meta Learner Search](https://arxiv.org/abs/1806.06927)] also suggests that approaches like MAML and its extensions tend to perform better with deeper neural architectures for few-shot classification tasks. While note-worthy, it nevertheless should be explored further with more experiments.

