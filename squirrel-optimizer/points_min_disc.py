######################################################################################
# Reads points minimum discrepancy to build a DataFrame for quick querying and mapping #
######################################################################################
import os
import sys
import typing
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

util_path = os.path.join('/'.join(__file__.split('/')[:-1]), 'utils')
sys.path.append(util_path)


class PointsMinDisc(AbstractOptimizer):
    def __init__(self, api_config, pop_size=24, warmstart=True, **kwargs):
        '''

        Parameters
        ----------
        warm_version : int,
            if 1, loads the incumbents collected initially by Diederick from Bayesmark tasks
            if 2, loads incumbents collected by Gresa from Bayesmark + OpenML datasets
        '''
        super(PointsMinDisc, self).__init__(api_config)
        self.api_config = api_config
        self.cs = PointsMinDisc.get_cs_dimensions(api_config)
        self.dimensions = len(self.cs.get_hyperparameters())
        self.pop_size = pop_size
        self.possible_sets = [8, 16, 24, 32, 40]
        self.warmstart = True
        self.warmstart_possible = True
        if self.warmstart:
            self.population, self.fitness = self._warmstart_init_population(self.pop_size,
                                                                            self.dimensions)
            self.iteration = 0
        else:
            self.population = self.init_population(self.pop_size)  # for initial iteration
            self.iteration = 0
        self.idxask = 0
        self.limit = self.pop_size // 8

    def read_local_warmstarts(self, output_format='array'):
        '''Function to read local files and output a set of incumbents in chosen format
        Parameters
        ----------
        path : str
            The directory that houses the csv file
        dimensions : The dimensionality of the parameter space
        size : int
            The number of configurations to suggest
        output_format : str
            If output_format='list-dict', return each config as dicts inside a list
            If output_format='list-list', return each config as a list inside a list
            If output_format='list-configspace', return each config as a ConfigSpace
                object inside a list
            If output_format='array', return each config as a numpy array inside a numpy array
            If output_format='dataframe', return each config as a row of a dataframe
        Returns
        -------
        incumbents : list/array/dataframe
            Containing 'size' (or lesser) incumbents sampled from local files
        '''
        self.pop_size = min([size if not size < self.pop_size else np.inf for size in self.possible_sets])
        path = "{}/points_minimum_discrepancy/{}_{}".format(util_path, self.pop_size, self.dimensions)
        df = pd.read_csv(filepath_or_buffer=path, sep=';', header=None, decimal=',').astype(float)
        incumbents = df.dropna().to_numpy().tolist()

        if output_format == 'list-dict':
            pass
        elif output_format == 'list-list':
            incumbents = pd.DataFrame(incumbents).to_numpy().tolist()
        elif output_format == 'list-configspace':
            cs = PointsMinDisc.get_cs_dimensions(self.api_config)
            incumbents = [self.vector_to_configspace(config, cs) for config in incumbents]
        elif output_format == 'array':
            incumbents = pd.DataFrame(incumbents).to_numpy()
        elif output_format == 'dataframe':
            incumbents = pd.DataFrame(incumbents)
        else:
            legal_formats = {"list-dict", "list-list", "list-configspace", "array", "dataframe"}
            raise ValueError(
                "Invalid output_format selected, "
                "'{}' not among {}".format(output_format, legal_formats)
            )
        return incumbents

    def _dict_to_configspace(self, config_dict):
        config = self.cs.sample_configuration()
        for param in config:
            if isinstance(self.cs.get_hyperparameter(param), UniformIntegerHyperparameter):
                config[param] = int(config_dict[param])
            elif isinstance(self.cs.get_hyperparameter(param), CategoricalHyperparameter):
                config[param] = self.cs.get_hyperparameter(param).choices[int(config_dict[param])]
            elif isinstance(self.cs.get_hyperparameter(param), OrdinalHyperparameter):
                config[param] = self.cs.get_hyperparameter(param).sequence[int(config_dict[param])]
            else:
                config[param] = config_dict[param]
        return config

    def _warmstart_init_population(self, n_suggestions, dimensions):
        fitness = [np.inf for i in range(n_suggestions)]

        if not os.path.isfile("{}/points_minimum_discrepancy/{}_{}".format(util_path, self.pop_size, self.dimensions)):
            population = self.init_population(n_suggestions)
            return population, fitness

        inc_list = self.read_local_warmstarts(output_format='list-dict')

        if len(inc_list) == 0:  # no match found --> new unseen parameter space
            self.warmstart_possible = False
            population = self.init_population(n_suggestions)
            return population, fitness

        ### reaching this part of the scope indicates api_config has partial or
        ###  complete overlaps from among the parameter spaces seen locally

        # converting configurations to [0,1] space for DE
        inc_list = np.array([
            self.boundary_check(np.array(config), fix_type='clip') for config in inc_list
        ])

        # if not enough incumbents extracted
        if len(inc_list) < n_suggestions:
            # initial remaining slots filled randomly as without warmstart
            extra_pop = self.init_population(n_suggestions - len(inc_list))
            inc_list = np.vstack((inc_list, extra_pop))

        # selecting only n_suggestions incumbents
        np.random.shuffle(inc_list)  # in-place shuffling
        population = inc_list[:n_suggestions]
        return population, fitness

    def init_population(self, pop_size=8):
        population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population


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

        # hp_list = []
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
                    log = True if param_space in ("log", "logit") else False
                    if param_type == "int":
                        hp = UniformIntegerHyperparameter(name=param_name, lower=param_range[0], upper=param_range[-1],
                                                          log=log)
                    else:
                        hp = UniformFloatHyperparameter(name=param_name, lower=param_range[0], upper=param_range[-1],
                                                        log=log)
            else:
                assert False, "type %s not handled in API" % param_type
            cs.add_hyperparameter(hp)

        return cs

    def init_with_rh(self, rh, **kwargs):
        pass

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
        next_guess = []
        for i in range(n_suggestions):
            next_guess.append(self.vector_to_configspace(self.population[(self.idxask*n_suggestions)+i]))
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

        self.iteration += 1
        self.idxask = (self.idxask + 1) % self.limit


if __name__ == "__main__":
    experiment_main(PointsMinDisc)
