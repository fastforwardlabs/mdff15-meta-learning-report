## Ethics

Meta-learning alleviates the need to collect vast amounts of data, and hence is applicable where supervised training examples are difficult (or 
even impossible) to acquire, given safety, security and privacy issues. If training efficient deep learning models is possible in such a scenario 
with just a handful of examples, it will benefit machine learning practitioners and its overall adoption.

Recent research^[[Fairness warnings and fair-MAML: learning fairly with minimal data](https://dl.acm.org/doi/abs/10.1145/3351095.3372839)] in fairness addresses the question of how a practitioner who has access to only a few labeled examples can successfully train a fair machine learning model. The paper suggests that one can do so by extending the MAML algorithm to Fair-MAML, such that each task includes a fairness regularization term in 
the task losses and a fairness hyperparameter—gamma—in hopes of encouraging MAML to learn generalizable internal representations that strike a desirable balance between accuracy and fairness. 
 
