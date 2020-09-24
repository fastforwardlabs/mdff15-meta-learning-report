_In early spring of 2019, we researched approaches that would allow a machine learning practitioner to perform supervised learning with only a 
limited number of examples available during training. This search led us to a new paradigm: meta-learning, in which an algorithm not only learns 
from a handful of examples, but also learns to classify novel classes during model inference. We decided to focus our research report—[Learning 
with Limited Labeled Data](https://blog.fastforwardlabs.com/2019/04/02/a-guide-to-learning-with-limited-labeled-data.html)—on active learning for 
deep neural networks, but we were both intrigued and fascinated with meta-learning as an emerging capability. This article is an attempt to throw 
some light on the great work that’s been done in this area so far._

## Introduction

Humans have an innate ability to learn new skills quickly. For example, we can look at one instance of a knife and be able to discriminate all knives from other cutlery items, like spoons and forks. Our ability to learn new skills and adapt to new environments quickly (based on only a few experiences or demonstrations) is not just limited to identifying new objects, learning a new language, or figuring out how to use a new tool;  our capabilities are much more varied. In contrast, machines—especially deep learning algorithms—typically learn quite differently. They require vast amounts of data and compute and may yet struggle to generalize. The reason humans are successful in adapting and learning quickly is that they leverage knowledge acquired from prior experience to solve novel tasks. In a similar fashion, meta-learning leverages previous knowledge acquired from data to solve novel tasks quickly and more efficiently.

![Figure 1: Humans can learn things quickly](figures/ff15-44.png)

### Why should we care?

An experienced ML practitioner might wonder, isn’t this covered by recent (and much-accoladed) advances in transfer learning? Well, no. Not exactly. 
First, supervised learning through deep learning methods requires massive amounts of labeled training data. These datasets are expensive to create, especially when one needs to involve a domain expert. While pre-training is beneficial, these approaches become less effective for domain-specific problems, which still require large amounts of task-specific labeled data to achieve good performance. 

In addition, certain real world problems have long-tailed and imbalanced data distributions, which may make it difficult to collect training 
examples.^[[Learning to Model the Tail (PDF)](https://papers.nips.cc/paper/7278-learning-to-model-the-tail.pdf)] For instance, in the case of 
search engines, perhaps a few keywords are commonly searched for, whereas a vast majority of keywords are rarely searched for. This may result in 
poor performance of models/applications based on long-tailed or imbalanced data distributions. The same could be true of recommendation engines; 
when there are not enough user reviews or ratings for obscure movies or products, it can 
hinder model performance.

![Figure 2: Long-tailed distributions](figures/ff15-45.png)

Most important, the ability to learn new tasks quickly during model inference is something that conventional machine learning approaches do not attempt. This is what makes meta-learning particularly attractive. 

### Why now?

From a deep learning perspective, meta-learning is particularly exciting and adoptable for three reasons: the ability to learn from a handful of examples, learning or adapting to novel tasks quickly, and the capability to build more generalizable systems. These are also some of the reasons why meta-learning is successful in applications that require data-efficient approaches; for example, robots are tasked with learning new skills in the real world, and are often faced with new environments.

Further, computer vision is one of the major areas in which meta-learning techniques have been explored to solve few-shot learning 
problems—including classification, object detection and segmentation, landmark prediction, video synthesis, and others.^[Meta-learning in Neural Networks: A Survey](https://arxiv.org/abs/2004.05439) Additionally, meta-learning has been popular in language modeling tasks, like filling in missing words^[Matching Networks for One-Shot Learning](https://arxiv.org/abs/1606.04080) and machine translation^[Meta-Learning for Low-Resource Neural Machine Translation](https://arxiv.org/abs/1808.08437), and is also being applied to speech recognition tasks, like cross-accent adaptation.^[Learning Fast Adaptation on Cross-Accented Speech Recognition](https://arxiv.org/abs/2003.01901)

![Figure 3: Applications - object detection, machine translation, missing words](figures/ff15-46.png)

As with any other machine learning capability that starts to show promise, there are now libraries and tooling that make meta-learning 
more accessible. Although not entirely production-ready, libraries like [torch-meta](https://github.com/tristandeleu/pytorch-meta), [learn2learn](https://github.com/learnables/learn2learn) and [meta-datasets](https://github.com/google-research/meta-dataset) help handle data, simplify processes when used with popular deep learning frameworks, and help document and benchmark performance on datasets. 

The rest of this report, along with its accompanying code, explores meta-learning, provides insight into how it works, and discusses its 
implications. We’ll do this using a simple, yet elegant algorithm—Model Agnostic Meta-Learning^[Model Agnostic Meta-learning for Fast Adaptation of Deep Networks (PDF)](https://arxiv.org/pdf/1703.03400.pdf)—applied to a few-shot classification problem, which was proposed a while ago, but 
continues to provide a good basis for extension and modification even today. 


