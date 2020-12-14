import typing
import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter

from smac.scenario.scenario import Scenario

from smac.stats.stats import Stats

from smac.initial_design import sobol_design, latin_hypercube_design, factorial_design, random_configuration_design, \
    default_configuration_design

from smac.utils.io.traj_logging import TrajLogger

from bayesmark.abstract_optimizer import AbstractOptimizer


class SMACInit(AbstractOptimizer):
    def __init__(self, api_config, config_space, lifetime):
        # The number of iterations that sobol(or LHD) sequence runs depends on which iteration when it is first initialized,
        # also this value depends on the number of the hyperparameters,
        # currently it ill run 2 or 3 iterations depending on the size of hyperparameter and init_design
        # lifetime: how many iterations smac_init will run
        super(SMACInit, self).__init__(api_config)
        self.cs = config_space
        self.num_hps = len(self.cs.get_hyperparameters())
        self.iteration = 0

        rng = np.random.RandomState(seed=0)
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                             'runcount-limit': 128,
                             "cs": self.cs,  # configuration space
                             "deterministic": True,
                             "limit_resources": False,
                             })

        self.stats = Stats(scenario)
        traj = TrajLogger(output_dir=None, stats=self.stats)

        number_of_hyperparameters = len(self.cs.get_hyperparameters())
        print('Found %d hyperparameters' % number_of_hyperparameters)
        bobo_initial_design_size = 8 * number_of_hyperparameters
        print(bobo_initial_design_size)
        initial_design_size = (bobo_initial_design_size // 8) * 8
        print(initial_design_size)
        initial_design_size = min(max(initial_design_size, 16), 8 * lifetime)
        print(initial_design_size)

        self.init_design_def_kwargs = {
                'cs': scenario.cs,  # type: ignore[attr-defined] # noqa F821
                'traj_logger': traj,
                'rng': rng,
                'ta_run_limit': scenario.ta_run_limit,  # type: ignore[attr-defined] # noqa F821
                'init_budget': initial_design_size,
        }

        self.initial_design = 'SOBOL'

        self.init_design_def_kwargs['init_budget'] = initial_design_size

        if self.initial_design == "DEFAULT":  # type: ignore[attr-defined] # noqa F821
            self.init_design_def_kwargs['max_config_fracs'] = 0.0
            initial_design_instance = default_configuration_design.DefaultConfiguration(**self.init_design_def_kwargs)
        elif self.initial_design == "RANDOM":  # type: ignore[attr-defined] # noqa F821
            self.init_design_def_kwargs['max_config_fracs'] = 0.0
            initial_design_instance = random_configuration_design.RandomConfigurations(**self.init_design_def_kwargs)
        elif self.initial_design == "LHD":  # type: ignore[attr-defined] # noqa F821
            initial_design_instance = latin_hypercube_design.LHDesign(**self.init_design_def_kwargs)
        elif self.initial_design == "SOBOL":  # type: ignore[attr-defined] # noqa F821
            initial_design_instance = sobol_design.SobolDesign(**self.init_design_def_kwargs)
        else:
            raise ValueError(self.initial_design)
        self.next_evaluations = initial_design_instance.select_configurations()

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
                    log = True if param_space in ("log", "logit") else False
                    if param_type == "int":
                        hp = UniformIntegerHyperparameter(name=param_name, lower=param_range[0], upper=param_range[-1],
                                                          log=log)
                    else:
                        hp = UniformFloatHyperparameter(name=param_name, lower=param_range[0], upper=param_range[-1],
                                                        log=log)
            else:
                assert False, "type %s not handled in API" % param_type
            hp_list.append(hp)
        cs.add_hyperparameters(hp_list)

        return cs

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

        next_guess = [{} for _ in range(n_suggestions)]
        for i in range(n_suggestions):
            eval_next = self.next_evaluations.pop(0)
            next_guess[i] = eval_next.get_dictionary()

        return next_guess

    def init_with_rh(self, rh, iteration):
        self.iteration = iteration

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary 使用where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        self.iteration += 1
