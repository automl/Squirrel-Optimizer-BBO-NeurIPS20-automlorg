import typing
import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark import np_util


class RandomOpt(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random):
        """Build wrapper class to use random search function in benchmark.
        Settings for `suggest_dict` can be passed using kwargs.
        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.cs = self.get_cs_dimensions(api_config)

    def suggest(self, n_suggestions=1):
        """Get suggestion.
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
        samples_config = self.cs.sample_configuration(size=n_suggestions)
        next_guess = [sample.get_dictionary() for sample in samples_config]
        return next_guess

    def init_with_rh(self, rh, **kwargs):
        # do nothing
        return

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
        pass

    def get_cs_dimensions(self, api_config: typing.Dict) -> ConfigurationSpace:
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
                    log = True if param_space in ("log", "logit") else False
                    if param_type == "int":
                        hp = UniformIntegerHyperparameter(name=param_name, lower=param_range[0],
                                                          upper=param_range[-1],
                                                          log=log)
                    else:
                        hp = UniformFloatHyperparameter(name=param_name, lower=param_range[0],
                                                        upper=param_range[-1],
                                                        log=log)
            else:
                assert False, "type %s not handled in API" % param_type
            hp_list.append(hp)
        cs.add_hyperparameters(hp_list)

        return cs
