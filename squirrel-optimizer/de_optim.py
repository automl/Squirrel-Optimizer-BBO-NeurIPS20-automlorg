import os
import sys
import typing
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

util_path = os.path.join('/'.join(__file__.split('/')[:-1]), 'utils')
sys.path.append(util_path)
from warmstart_helper import warmstart_load


class DEOptimizer(AbstractOptimizer):
    def __init__(self, api_config, pop_size=8, max_age=None, mutation_factor=0.5,
                 crossover_prob=0.5, budget=None, strategy='best2_bin', f_adaptation="SinDE",
                 sin_de_configuration=1, warmstart=False, **kwargs):
        super(DEOptimizer, self).__init__(api_config)
        self.api_config = api_config
        self.cs = DEOptimizer.get_cs_dimensions(api_config)
        self.dimensions = len(self.cs.get_hyperparameters())
        self.sin_de_configuration = sin_de_configuration
        # DE related variables
        self.initial_mutation = mutation_factor
        self.pop_size = pop_size
        self.max_age = max_age
        self.mutation_factor = [mutation_factor] * self.pop_size
        self.crossover_prob = crossover_prob
        self.mutation_strategy = strategy.split('_')[0]
        self.crossover_strategy = strategy.split('_')[1]
        self.budget = budget
        self.f_adaptation = self.get_f_adapt(f_adaptation)

        # Miscellaneous
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else ''

        # DE specific
        self.freq = 0.25
        self.iter_max = 16
        self.warmstart = warmstart
        if self.warmstart:
            self.population, self.fitness = self._warmstart_init_population(self.pop_size)
            self.iteration = 0
        else:
            self.population = self.init_population(self.pop_size)  # for initial iteration
            self.fitness = np.inf * np.ones(self.pop_size)  # unevaluated individuals
            self.iteration = 1

        self.trial_population = self.population
        self.age = np.zeros(self.pop_size)
        self.history = []
        self.traj = []
        self.runtime = []

    def get_f_adapt(self, f_adaptation="Const"):
        if f_adaptation == "Const":
            return self.const_adapt
        if f_adaptation == "SaMDE":
            return self.sa_mde
        if f_adaptation == "SinDE":
            return self.sin_de
        return self.const_adapt

    def const_adapt(self):
        return [self.initial_mutation] * self.pop_size

    def sin_de(self, configuration=1):
        if configuration == 1:
            F_it = 1 / 2 * (np.sin(2 * np.pi * self.freq * self.iteration) * self.iteration / self.iter_max + 1)
            Cr = 0.5
        elif configuration == 2:
            F_it = 1 / 2 * (np.sin(2 * np.pi * self.freq * self.iteration) * self.iteration / self.iter_max + 1)
            Cr = 1 / 2 * (np.sin(2 * np.pi * self.freq * self.iteration) * self.iteration / self.iter_max + 1)
        elif configuration == 3:
            F_it = 1 / 2 * (np.sin(2 * np.pi * self.freq * self.iteration) * (self.iter_max - self.iteration)
                            / self.iter_max + 1)
            Cr = 1 / 2 * (np.sin(2 * np.pi * self.freq * self.iteration) * (self.iter_max - self.iteration)
                          / self.iter_max + 1)
        elif configuration == 4:
            F_it = 1 / 2 * (np.sin(2 * np.pi * self.freq * self.iteration) * (self.iter_max - self.iteration)
                            / self.iter_max + 1)
            Cr = 0.9
        elif configuration == 5:
            F_it = 0.5
            Cr = 1 / 2 * (np.sin(2 * np.pi * self.freq * self.iteration) * self.iteration / self.iter_max + 1)
        elif configuration == 6:
            F_it = 0.5
            Cr = 1 / 2 * (np.sin(2 * np.pi * self.freq * self.iteration) * (self.iter_max - self.iteration)
                          / self.iter_max + 1)
        return [F_it] * self.pop_size, Cr

    def _dict_to_configspace(self, config_dict):
        config = self.cs.sample_configuration()
        try:
            for param in config:
                if isinstance(self.cs.get_hyperparameter(param), UniformIntegerHyperparameter):
                    config[param] = int(config_dict[param])
                elif isinstance(self.cs.get_hyperparameter(param), CategoricalHyperparameter):
                    if self.api_config[param]['type'] == 'bool':
                        # config_dict[param] will either be {0, 1} or {True, False}
                        # int() type cast should work fine
                        config[param] = \
                            self.cs.get_hyperparameter(param).choices[int(config_dict[param])]
                    else:
                        # simply reassign the string
                        # assumption: a valid string from the 'choices' the parameter can take
                        config[param] = config_dict[param]
                else:
                    config[param] = config_dict[param]
        except Exception as e:
            print(e)
        return config

    def _dict_to_vector(self, config_dict):
        config = self._dict_to_configspace(config_dict)
        vector = self.configspace_to_vector(config)
        return vector

    def _sanitize_null_configs(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data).dropna()
            data = [data.loc[i].to_dict() for i in data.index]
            return data
        return data.dropna()

    def _warmstart_init_population(self, n_suggestions):
        fitness = [np.inf for i in range(n_suggestions)]

        if not os.path.isfile("{}/df.pkl".format(util_path)):
            population = self.init_population(n_suggestions)
            return population, fitness

        # Load local incumbent info
        with open("{}/df.pkl".format(util_path), "rb") as f:
            self.param_spaces = pickle.load(f)

        # Extract incumbents based on how much api_config matches the known spaces
        inc_list = warmstart_load(
            self.param_spaces, self.api_config, path="{}/param_spaces".format(util_path)
        )
        inc_list = self._sanitize_null_configs(inc_list)  # remove configs with nan

        if len(inc_list) == 0:  # no match found --> new unseen parameter space
            population = self.init_population(n_suggestions)
            return population, fitness

        ### this part of the scope indicates api_config has partial or complete overlaps
        ###  from among the parameter spaces seen locally

        # converting configurations to [0,1] space for DE
        inc_list = np.array([
            self.boundary_check(
                self.configspace_to_vector(self._dict_to_configspace(config)), fix_type='clip'
            ) for config in inc_list
        ])

        if len(inc_list) < n_suggestions:  # not enough incumbents extracted
            # initial remaining slots randomly as without warmstart
            extra_pop = self.init_population(n_suggestions - len(inc_list))
        else:
            inc_list = inc_list[
                np.random.choice(np.arange(inc_list.shape[0]),
                                 size=int(n_suggestions / 2), replace=False)
            ]
            extra_pop = self.init_population(n_suggestions - int(n_suggestions / 2))
        inc_list = np.vstack((inc_list, extra_pop))

        population = inc_list[
            np.random.choice(np.arange(inc_list.shape[0]), size=8, replace=False)
        ]
        return population, fitness

    def init_population(self, pop_size=8):
        population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population

    def sample_population(self, size=3):
        selection = np.random.choice(np.arange(len(self.population)), size, replace=False)
        return np.take(self.population, selection)

    def sa_mde(self, tuned=False):
        if tuned:
            mean = self.mean
            std = self.std
        else:
            mean = 0.5
            std = 0.3
        F = np.random.normal(mean, std, self.pop_size)
        return F

    def boundary_check(self, vector, fix_type='random'):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.
        Parameters
        ----------
        vector : array
            The vector describing the individual from the population
        fix_type : str, {'random', 'clip'}
            if 'random', the values are replaced with a random sampling from [0,1)
            if 'clip', the values are clipped to the closest limit from {0, 1}
        Returns
        -------
        array
        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    def vector_to_configspace(self, vector):
        '''Converts numpy array to ConfigSpace object
        Works when self.cs is a ConfigSpace object and the input vector is in the domain [0, 1].
        '''
        new_config = self.cs.sample_configuration()
        for i, hyper in enumerate(self.cs.get_hyperparameters()):
            if type(hyper) == OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1 / len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1 / len(hyper.choices))
                param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.clip(  # clipping to handle precision leaks over bounds
                        np.exp(np.log(hyper.lower) + vector[i] * log_range),
                        hyper.lower, hyper.upper
                    )
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                if type(hyper) == UniformIntegerHyperparameter:
                    param_value = np.round(param_value).astype(int)  # converting to discrete (int)
            new_config[hyper.name] = param_value
        return new_config

    def configspace_to_vector(self, config):
        '''Converts ConfigSpace object to numpy array scaled to [0,1]
        Works when self.cs is a ConfigSpace object and the input config is a ConfigSpace object.
        '''
        dimensions = len(config.keys())
        vector = [0.0 for i in range(dimensions)]

        for i, name in enumerate(config):
            hyper = self.cs.get_hyperparameter(name)
            if type(hyper) == OrdinalHyperparameter:
                nlevels = len(hyper.sequence)
                vector[i] = hyper.sequence.index(config[name]) / nlevels
            elif type(hyper) == CategoricalHyperparameter:
                nlevels = len(hyper.choices)
                vector[i] = hyper.choices.index(config[name]) / nlevels
            else:
                bounds = (hyper.lower, hyper.upper)
                param_value = config[name]
                if hyper.log:
                    vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                else:
                    vector[i] = (config[name] - bounds[0]) / (bounds[1] - bounds[0])
        return np.array(vector)

    def mutation_rand1(self, r1, r2, r3, idx):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor[idx] * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5, idx):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor[idx] * diff1 + self.mutation_factor[idx] * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2, idx):
        '''Performs the 'current-to-best' type of DE mutation
        '''
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor[idx] * diff1 + self.mutation_factor[idx] * diff2
        return mutant

    def mutation_rand2dir(self, r1, r2, r3, idx):
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor[idx] * diff / 2
        return mutant

    def mutation(self, current=None, best=None, idx=0):
        '''Performs DE mutation based on the strategy
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand1(r1, r2, r3, idx)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5, idx)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand2dir(r1, r2, r3, idx)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2, idx)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4, idx)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2, idx)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3, idx)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = np.random.rand(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''Performs the exponential crossover of DE
        '''
        n = np.random.randint(0, self.dimensions)
        L = 0
        while ((np.random.rand() < self.crossover_prob) and L < self.dimensions):
            idx = (n + L) % self.dimensions
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''Performs DE crossover based on the strategy
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        else:
            offspring = self.crossover_exp(target, mutant)
        return offspring

    @staticmethod
    def get_cs_dimensions(api_config: typing.Dict) -> ConfigurationSpace:
        """
        Help routine to setup ConfigurationSpace search space in constructor.
        Take api_config as argument so this can be static.
        Parameters
        ----------
        api_config: Dict
            api dictionary to construct
        Returns
        -------
        cs: ConfigurationSpace
            ConfigurationSpace that contains the same hyperparameter as api_config
        """
        # TODO 2 options to transform the real and int hyperaparameters in different scales
        #  option 1: similar to example_submission.skopt.optimizer, merge 'logit' into 'log' and 'bilog' into 'linear'
        #  option 2: use the api bayesmark.space.space to warp and unwarp the samples
        cs = ConfigurationSpace()
        param_list = sorted(api_config.keys())

        hp_list = []
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]
            param_space = param_config.get("space", None)
            param_values = param_config.get("values", None)
            param_range = param_config.get("range", None)

            if param_type == "cat":
                assert param_space is None
                assert param_range is None
                hp = CategoricalHyperparameter(name=param_name, choices=param_values)
            elif param_type == "bool":
                assert param_space is None
                assert param_values is None
                assert param_range is None
                hp = CategoricalHyperparameter(name=param_name, choices=[True, False])
            elif param_type == "ordinal":
                # appear in example_submission.skopt.optimizer but not in README
                assert param_space is None
                assert param_range is None
                hp = OrdinalHyperparameter(name=param_name, sequence=param_values)
            elif param_type in ("int", "real"):
                if param_values is not None:
                    # TODO: decide whether we treat these parameters as discrete values
                    #  or step function (example see example_submission.skopt.optimizer, line 71-77)
                    # sort the values to store them in OrdinalHyperparameter
                    param_values_sorted = np.sort(param_values)
                    hp = OrdinalHyperparameter(name=param_name, sequence=param_values_sorted)
                else:
                    log = True if param_space == "log" else False

                    if param_type == "int":
                        hp = UniformIntegerHyperparameter(name=param_name,
                                                          lower=param_range[0],
                                                          upper=param_range[-1],
                                                          log=log)
                    else:
                        hp = UniformFloatHyperparameter(name=param_name, lower=param_range[0],
                                                        upper=param_range[-1], log=log)
            else:
                assert False, "type %s not handled in API" % param_type
            hp_list.append(hp)
        cs.add_hyperparameters(hp_list)
        return cs

    def get_individuals_from_ranking(self, configs, fitness):

        inc_idx = np.argmin(fitness)
        best_conf, best_y = self._dict_to_vector(configs[inc_idx]), fitness[inc_idx]
        fitness.pop(inc_idx)
        configs.pop(inc_idx)

        rankings = rankdata(fitness, method='min')
        sum_rankings = np.sum(rankings)
        n = len(fitness)
        weights = [(sum_rankings - rank)/((n-1)*sum_rankings) for rank in rankings]
        idx = np.random.choice(np.arange(len(fitness)), p=weights, size=self.pop_size-1, replace=False)
        try:
            self.population = [self._dict_to_vector(conf_dict) for conf_dict in np.take(configs, idx)] + [best_conf]
            self.fitness = list(np.take(fitness, idx)) + [best_y]
        except Exception as e:
            print("Error in extraction")
            print(e)


    def init_with_rh(self, rh, iteration=0, return_vector=True, choice='rand'):
        # Get 8 configs to use for warmstart
        try:
            configs_all = np.array([value[0] for value in rh])
            y_all = np.array([value[1] for value in rh])
            if len(configs_all) > 7:
                if choice =='best':
                    sorted_idx = np.argsort(y_all)
                    if return_vector:
                        vectors = [self._dict_to_vector(conf_dict) for conf_dict in
                                   np.take(configs_all, sorted_idx[:8])]
                        self.population, self.fitness = vectors, np.take(y_all, sorted_idx[:8])
                elif choice =='rand':
                    configs_SMAC = configs_all[-5 * self.pop_size:]
                    y_SMAC = y_all[-5 * self.pop_size:]
                    inc_idx = np.argmin(y_all)
                    best_conf, best_y = self._dict_to_vector(configs_all[inc_idx]), y_all[inc_idx]
                    rand_idx = np.random.choice(np.arange(len(configs_SMAC)), self.pop_size - 1, replace=False)
                    if return_vector:
                        vectors = [self._dict_to_vector(conf_dict) for conf_dict in np.take(configs_SMAC, rand_idx)]
                        self.population = vectors + [best_conf]
                        self.fitness = list(np.take(y_SMAC, rand_idx)) + [best_y]
                else:
                    self.get_individuals_from_ranking(list(configs_all), list(y_all))
            self.iteration = iteration
        except Exception as e:
            self.population = self.init_population(pop_size=self.pop_size)

    def suggest(self, n_suggestions: int = 1) -> typing.List[typing.Dict]:
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        try:
            if self.iteration == 0:
                next_guess = [self.vector_to_configspace(trial) for trial in self.population]
                return next_guess
            self.n_suggestions = n_suggestions
            trials = []
            self.mutation_factor, _ = self.f_adaptation(configuration=self.sin_de_configuration)
            self.crossover_prob = np.random.normal(loc=0.5, scale=0.1)
            idxbest = np.argmin(self.fitness)
            best = self.population[idxbest]
            for j in range(self.pop_size):
                target = self.population[j]
                donor = self.mutation(current=target, best=best, idx=j)
                trial = self.crossover(target, donor)
                trial = self.boundary_check(trial)
                trials.append(trial)
            trials = np.array(trials)
        except Exception as e:
            trials = self.init_population(pop_size=self.pop_size)
        self.trial_population = trials  # [0,1] representation of next_guess seen in observe()
        next_guess = [self.vector_to_configspace(trial) for trial in trials]
        return next_guess

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        trials = self.trial_population

        # only the new points (X) from suggest() and its evaluations (y) are passed as input
        for i in range(len(y)):
            # evaluation of the newly created individuals
            fitness = y[i]
            # selection -- competition between parent[i] -- child[i]
            ## equality is important for landscape exploration
            if fitness <= self.fitness[i]:
                self.population[i] = trials[i]
                self.fitness[i] = fitness
        self.iteration += 1


if __name__ == "__main__":
    experiment_main(DEOptimizer)
