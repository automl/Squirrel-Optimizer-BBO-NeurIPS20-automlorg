# Squirrel: A Switching Hyperparameter Optimizer

Motivated by the fact that different optimizers work well on different problems, our approach switches between different optimizers. Since the team names on the competition's leaderboard were randomly generated, consisting of an adjective and an animal with the same initial letter, we called our approach the Switching Squirrel, short, Squirrel.

In our Squirrel framework, we switched between the following components: 
1. An initial design (for known hyperparameter spaces: found by meta-learning; otherwise: selected by differential evolution) (3 batches);
2. Optimization using Bayesian optimization by integrating the SMAC optimizer with a portfolio of different triplets of surrogate model, acquisition function, and output space transformation (8 batches); and
3. Optimization using Differential Evolution with parameter adaptation (5 batches)  

## Results 
Our Squirrel **ranked 3rd** with a **score of 92.551** on [offical learderboard](https://bbochallenge.com/leaderboard), and also won **1st place** in [alternate leaderboard](https://bbochallenge.com/altleaderboard) (with a score of **94.845476** and the organizers' bootstrap analysis showing a 100% confidence in this 1st place ranking). 


## Run Squirrel locally
We used the [Bayesmark](https://github.com/uber/bayesmark) benchmark framework for the local experiments with Squirrel. See the Bayesmark [documentation](https://bayesmark.readthedocs.io/en/latest/) for the details.
##### Create and activate virtual environment
```console
> python3 -m venv venv  # Please use Python 3.6.10.
> source venv/bin/activate
```
##### Install requirements
```console
> pip install -r environment.txt -r squirrel-optimizer/requirements.txt
```
##### Run Squirrel on Bayesmark
```console
>  ./run_local.sh squirrel-optimizer/ 3
...
--------------------
Final score `100 x (1-loss)` for leaderboard:
optimizer
squirrel-optimizer_0.0.6_6434ac2    102.238945
```

## Team Members
* Noor Awad
* Gresa Shala 
* Difan Deng
* Neeratyoy Mallik
* Matthias Feurer
* Katharina Eggensperger
* Andre' Biedenkapp
* Diederick Vermetten
* Hao Wang
* Carola Doerr
* Marius Lindauer
* Frank Hutter

## License
Our implementation is released under Apache License 2.0.
