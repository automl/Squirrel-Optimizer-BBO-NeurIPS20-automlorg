Squirrel: A Switching Hyperparameter Optimizer

Motivated by the fact that different optimizers work well on different problems, our approach switches between different optimizers. Since the team names on the competition's leaderboard were randomly generated, consisting of an adjective and an animal with the same initial letter, we called our approach the Switching Squirrel, short, Squirrel.

In our Squirrel framework, we switched between the following components: 
1. An initial design (for known hyperparameter spaces: found by meta-learning; otherwise: selected by differential evolution) (3 batches);
2. Optimization using Bayesian optimization by integrating the SMAC optimizer with a portfolio of different triplets of surrogate model, acquisition function, and output space transformation (8 batches); and
3. Optimization using Differential Evolution with parameter adaptation (5 batches)   
