import typing, copy
import numpy as np
import pandas as pd


from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark import np_util

# Import other optimizers here
from init_design import InitialDesign as INIT
from points_min_disc import PointsMinDisc
from smac_optim import SMAC4EPMOpimizer as SMAC
from de_optim import DEOptimizer as DE
from cma_es_optim import CMA
from smac_init_optim import SMACInit

from utils import TRANS, INV_TRANS

# We don't need this
# from random_search import RandomOpt
# from cma_es import _CMA


class SwitchingOptimizer(AbstractOptimizer):
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

        self.random = random
        self._api_config = copy.deepcopy(api_config) # keep the original values for later usage
        self.api_config, self._par = self._parse_api_config(api_config)
        self.cs = self.get_cs_dimensions(self.api_config)
        self.original_cs = self.get_cs_dimensions(self._api_config)
        print("ConfigSpace: ")
        print(self.cs)
        print("Original ConfigSpace: ")
        print(self.original_cs)
        self.optimizer_dict = {
            # TODO: Think about whether we can/want to rely on '8' here
            # For debugging, you can replace any of the following with this opt
            # [RandomOpt, {"api_config": self.api_config}]
            # NOTE: all the optimizers should take `self.cs` and `self.api_config` to initialize
            "Warmstart": [INIT, {"api_config": self._api_config, "pop_size": 24, "warmstart": True}],
            "SOBOL": [SMACInit, {"api_config": self.api_config, "config_space": self.cs, 'lifetime': 3}],
            "PointsMD": [PointsMinDisc, {"api_config": self.api_config, "popsize": 24}],
            "SMAC": [SMAC, {"api_config": self.api_config, "config_space": self.cs, "parallel_setting": 'KB'}],
            "DE": [DE, {"api_config": self.api_config, "pop_size": 8, "max_age": None,
                        "mutation_factor": 0.5, "crossover_prob": 0.5, "budget": None,
                        "strategy": 'best2_bin', "f_adaptation": "SinDE", "warmstart": False,
                        "sin_de_configuration": 2}],
                        "cma": [CMA, {"cs": self.cs, "popsize": 8, "lb": 0, "ub": 1}],
        }
        self.default_opt = "DE"
        self.last = None
        self.cur_opt = None
        self.rh = []
        self.Q_table = None
        self._num_iters = 0
        self.cur_opt_str = ""
        self._max_allowed_switches = 3
        # The schedule has to probably be hard-coded here since we are not sure if we can load it somehow
        # Hand designed Schedule for testing:
        self.Q_table, self._max_allowed_switches = self._fixed_policy_warm_smac8_de5_or_de4_smac8_de5()

        num_iter_smac_init = 0

        for i in self.Q_table:
            self.Q_table[i] = np.array(self.Q_table[i])
            num_iter_smac_init += 1 if self.Q_table[i][1] != 0 else 0

        self.optimizer_dict["SOBOL"][1]['lifetime'] = num_iter_smac_init

    def _parse_api_config(self, api_config):
        _par = {}
        api_config = copy.deepcopy(api_config)
        for param_name in sorted(api_config.keys()):
            param_config = api_config[param_name]
            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)

            # conver the search parameter range using the `bilog` or `logit` transformation
            if param_space in ['bilog', 'logit']:
                if 'values' in param_config:
                    # parameters with 'values' specified are treated as ordinals
                    # transformations are not applicable
                    continue
                param_config['range'] = list(map(TRANS[param_space], param_range))
                _par[param_name] = param_space
                param_config['space'] = 'linear'
                param_config['type'] = 'real'
        return api_config, _par

    def _default_policy(self):
        # First & second iteration is warmstart (INIT)
        # Third iteration switches to SMAC (switch = 1)
        # Seventh iteration switches to DE (switch = 2 [max switches allowed])
        # CMA never switched to since max 2 switches allowed (?)
        Q_table = {
            (0,): [1, 0, 0, 0, 0, 0],
            (1,): [1, 0, 0, 0, 0, 0],
            (2,): [0, 1, 0, 0, 0, 0],
            (3,): [0, 1, 0, 0, 0, 0],
            (4,): [0, 0, 0, 1, 0, 0],
            (5,): [0, 0, 0, 1, 0, 0],
            (6,): [0, 0, 0, 0, 1, 0],
            (7,): [0, 0, 0, 0, 1, 0],
            (8,): [0, 0, 0, 0, 1, 0],
            (9,): [0, 0, 0, 0, 1, 0],
            (10,): [0, 0, 0, 0, 1, 0],
            (12,): [0, 0, 0, 0, 0, 1],
            (13,): [0, 0, 0, 0, 0, 1],
            (14,): [0, 0, 0, 0, 0, 1],
            (15,): [0, 0, 0, 0, 0, 1],
            (16,): [0, 0, 0, 0, 0, 1],
        }
        _max_allowed_switches = 3
        return Q_table, _max_allowed_switches

    def _fixed_policy_pointsmd_smac(self):
        # First & second iteration is warmstart (INIT)
        # Third iteration switches to SMAC (switch = 1)
        # Seventh iteration switches to DE (switch = 2 [max switches allowed])
        # CMA never switched to since max 2 switches allowed (?)
        Q_table = {
            (0,): [0, 0, 1, 0, 0, 0],
            (1,): [0, 0, 1, 0, 0, 0],
            (2,): [0, 0, 1, 0, 0, 0],
            (3,): [0, 0, 1, 0, 0, 0],
            (4,): [0, 0, 0, 1, 0, 0],
            (5,): [0, 0, 0, 1, 0, 0],
            (6,): [0, 0, 0, 1, 0, 0],
            (7,): [0, 0, 0, 1, 0, 0],
            (8,): [0, 0, 0, 1, 0, 0],
            (9,): [0, 0, 0, 1, 0, 0],
            (10,): [0, 0, 0, 1, 0, 0],
            (12,): [0, 0, 0, 1, 0, 0],
            (13,): [0, 0, 0, 1, 0, 0],
            (14,): [0, 0, 0, 1, 0, 0],
            (15,): [0, 0, 0, 1, 0, 0],
            (16,): [0, 0, 0, 1, 0, 0],
        }
        _max_allowed_switches = 1
        return Q_table, _max_allowed_switches

    def _fixed_policy_warm_smac8_cma5(self):
        ### Schedule slightly different from _default_policy() in terms of iterations allotted
        # First iteration is warmstart (INIT)
        # Second iteration switches to SMAC (switch = 1)
        # Tenth iteration switches to DE (switch = 2 [max switches allowed])
        Q_table = {
            (0,): [1, 0, 0, 0, 0, 0],
            (1,): [1, 0, 0, 0, 0, 0],
            (2,): [1, 0, 0, 0, 0, 0],
            (3,): [0, 0, 0, 1, 0, 0],
            (4,): [0, 0, 0, 1, 0, 0],
            (5,): [0, 0, 0, 1, 0, 0],
            (6,): [0, 0, 0, 1, 0, 0],
            (7,): [0, 0, 0, 1, 0, 0],
            (8,): [0, 0, 0, 1, 0, 0],
            (9,): [0, 0, 0, 1, 0, 0],
            (10,): [0, 0, 0, 1, 0, 0],
            (11,): [0, 0, 0, 0, 0, 1],
            (12,): [0, 0, 0, 0, 0, 1],
            (13,): [0, 0, 0, 0, 0, 1],
            (14,): [0, 0, 0, 0, 0, 1],
            (15,): [0, 0, 0, 0, 0, 1],
        }
        _max_allowed_switches = 2
        return Q_table, _max_allowed_switches

    def _fixed_policy_warm_smac8_de5_or_de4_smac8_de5(self):

        o, kw = self.optimizer_dict["Warmstart"]
        self.cur_opt = o(**kw)
        # This is used in cases when we use warmstart for known spaces and another policy for unknown spaces
        # If warmstart isn't possible, we change the policy to
        if not self.cur_opt.warmstart_possible:
            Q_table, _max_allowed_switches = self._fixed_policy_warm_de3_smac7_de5()
        else:
            Q_table, _max_allowed_switches = self._fixed_policy_warm_smac8_de5()

        return Q_table, _max_allowed_switches

    def _fixed_policy_warm_smac8_de5(self):
        ### Schedule slightly different from _default_policy() in terms of iterations allotted
        # First iteration is warmstart (INIT)
        # Second iteration switches to SMAC (switch = 1)
        # Tenth iteration switches to DE (switch = 2 [max switches allowed])
        Q_table = {
            (0,): [1, 0, 0, 0, 0, 0],
            (1,): [1, 0, 0, 0, 0, 0],
            (2,): [1, 0, 0, 0, 0, 0],
            (3,): [0, 0, 0, 1, 0, 0],
            (4,): [0, 0, 0, 1, 0, 0],
            (5,): [0, 0, 0, 1, 0, 0],
            (6,): [0, 0, 0, 1, 0, 0],
            (7,): [0, 0, 0, 1, 0, 0],
            (8,): [0, 0, 0, 1, 0, 0],
            (9,): [0, 0, 0, 1, 0, 0],
            (10,): [0, 0, 0, 1, 0, 0],
            (11,): [0, 0, 0, 0, 1, 0],
            (12,): [0, 0, 0, 0, 1, 0],
            (13,): [0, 0, 0, 0, 1, 0],
            (14,): [0, 0, 0, 0, 1, 0],
            (15,): [0, 0, 0, 0, 1, 0],
        }
        _max_allowed_switches = 2
        return Q_table, _max_allowed_switches

    def _fixed_policy_warm_de3_smac7_de5(self):
        ### Schedule slightly different from _default_policy() in terms of iterations allotted
        # First iteration is warmstart (INIT)
        # Second iteration switches to SMAC (switch = 1)
        # Tenth iteration switches to DE (switch = 2 [max switches allowed])
        Q_table = {
            (0,): [1, 0, 0, 0, 0, 0],
            (1,): [0, 0, 0, 0, 1, 0],
            (2,): [0, 0, 0, 0, 1, 0],
            (3,): [0, 0, 0, 1, 0, 0],
            (4,): [0, 0, 0, 1, 0, 0],
            (5,): [0, 0, 0, 1, 0, 0],
            (6,): [0, 0, 0, 1, 0, 0],
            (7,): [0, 0, 0, 1, 0, 0],
            (8,): [0, 0, 0, 1, 0, 0],
            (9,): [0, 0, 0, 1, 0, 0],
            (10,): [0, 0, 0, 1, 0, 0],
            (11,): [0, 0, 0, 0, 1, 0],
            (12,): [0, 0, 0, 0, 1, 0],
            (13,): [0, 0, 0, 0, 1, 0],
            (14,): [0, 0, 0, 0, 1, 0],
            (15,): [0, 0, 0, 0, 1, 0],
        }
        _max_allowed_switches = 3
        return Q_table, _max_allowed_switches

    def _fixed_policy_only_warm(self):
        # warmstart and run only SMAC
        # only 1 switch
        Q_table = {
            (0,): [1, 0, 0, 0, 0, 0],
            (1,): [1, 0, 0, 0, 0, 0],
            (2,): [1, 0, 0, 0, 0, 0],
            (3,): [1, 0, 0, 0, 0, 0],
            (4,): [1, 0, 0, 0, 0, 0],
            (5,): [1, 0, 0, 0, 0, 0],
            (6,): [1, 0, 0, 0, 0, 0],
            (7,): [1, 0, 0, 0, 0, 0],
            (8,): [1, 0, 0, 0, 0, 0],
            (9,): [1, 0, 0, 0, 0, 0],
            (10,): [1, 0, 0, 0, 0, 0],
            (12,): [1, 0, 0, 0, 0, 0],
            (13,): [1, 0, 0, 0, 0, 0],
            (14,): [1, 0, 0, 0, 0, 0],
            (15,): [1, 0, 0, 0, 0, 0],
            (16,): [1, 0, 0, 0, 0, 0],
        }
        _max_allowed_switches = 1
        return Q_table, _max_allowed_switches

    def _fixed_policy_only_smac(self):
        # warmstart and run only SMAC
        # only 1 switch
        Q_table = {
            (0,): [1, 0, 0, 0, 0, 0],
            (1,): [0, 1, 0, 0, 0, 0],
            (2,): [0, 1, 0, 0, 0, 0],
            (3,): [0, 1, 0, 0, 0, 0],
            (4,): [0, 0, 0, 1, 0, 0],
            (5,): [0, 0, 0, 1, 0, 0],
            (6,): [0, 0, 0, 1, 0, 0],
            (7,): [0, 0, 0, 1, 0, 0],
            (8,): [0, 0, 0, 1, 0, 0],
            (9,): [0, 0, 0, 1, 0, 0],
            (10,): [0, 0, 0, 1, 0, 0],
            (12,): [0, 0, 0, 1, 0, 0],
            (13,): [0, 0, 0, 1, 0, 0],
            (14,): [0, 0, 0, 1, 0, 0],
            (15,): [0, 0, 0, 1, 0, 0],
            (16,): [0, 0, 0, 1, 0, 0],
        }
        _max_allowed_switches = 2
        return Q_table, _max_allowed_switches

    def _fixed_policy_only_de(self):
        # warmstart and run only DE
        # only 1 switch
        Q_table = {
            (0,): [0, 0, 0, 0, 1, 0],
            (1,): [0, 0, 0, 0, 1, 0],
            (2,): [0, 0, 0, 0, 1, 0],
            (3,): [0, 0, 0, 0, 1, 0],
            (4,): [0, 0, 0, 0, 1, 0],
            (5,): [0, 0, 0, 0, 1, 0],
            (6,): [0, 0, 0, 0, 1, 0],
            (7,): [0, 0, 0, 0, 1, 0],
            (8,): [0, 0, 0, 0, 1, 0],
            (9,): [0, 0, 0, 0, 1, 0],
            (10,): [0, 0, 0, 0, 1, 0],
            (12,): [0, 0, 0, 0, 1, 0],
            (13,): [0, 0, 0, 0, 1, 0],
            (14,): [0, 0, 0, 0, 1, 0],
            (15,): [0, 0, 0, 0, 1, 0],
            (16,): [0, 0, 0, 0, 1, 0],
        }
        _max_allowed_switches = 1
        return Q_table, _max_allowed_switches

    def _fixed_policy_only_cma(self):
        # warmstart and run only CMA-ES
        # only 1 switch
        Q_table = {
            (0,): [1, 0, 0, 0, 0, 0],
            (1,): [1, 0, 0, 0, 0, 0],
            (2,): [1, 0, 0, 0, 0, 0],
            (3,): [0, 0, 0, 0, 0, 1],
            (4,): [0, 0, 0, 0, 0, 1],
            (5,): [0, 0, 0, 0, 0, 1],
            (6,): [0, 0, 0, 0, 0, 1],
            (7,): [0, 0, 0, 0, 0, 1],
            (8,): [0, 0, 0, 0, 0, 1],
            (9,): [0, 0, 0, 0, 0, 1],
            (10,): [0, 0, 0, 0, 0, 1],
            (12,): [0, 0, 0, 0, 0, 1],
            (13,): [0, 0, 0, 0, 0, 1],
            (14,): [0, 0, 0, 0, 0, 1],
            (15,): [0, 0, 0, 0, 0, 1],
            (16,): [0, 0, 0, 0, 0, 1],
        }
        _max_allowed_switches = 1
        return Q_table, _max_allowed_switches

    def get_meta_features(self):
        # do something with self.cs
        cats = 0
        ints = 0
        ord = 0
        fls = 0
        for i in self.cs.get_hyperparameters():
            if isinstance(i, UniformFloatHyperparameter):
                fls += 1
            elif isinstance(i, CategoricalHyperparameter):
                cats += 1
            elif isinstance(i, OrdinalHyperparameter):
                ord += 1
            elif isinstance(i, UniformIntegerHyperparameter):
                ints += 1
        return [cats, ints, ord, fls]

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
        # TODO: Check that these if-else statements work for all cases
        print("\nBEGINNING SUGGEST(): ")
        if self._max_allowed_switches > 0:
            # Compute whether we need to switch
            # TODO make this an actual state, for now just switch wrt #iter, see line 54++
            # state = tuple([self._num_iters] + self.get_meta_features())

            state = tuple([self._num_iters])
            # Set next_opt to default opt, in case state is not in Q table
            next_opt = self.default_opt
            if state in self.Q_table:
                action = self.Q_table[state].argmax()
                next_opt = list(self.optimizer_dict.keys())[action]

            if next_opt == self.cur_opt_str:
                pass
                # do nothing since we continue using the same optimizer
            else:
                # Initialize new optimizer

                # if we are on the first iteration, don't decrease max_switches
                if self._num_iters != 0:
                    self._max_allowed_switches -= 1

                # init optimizer
                o, kw = self.optimizer_dict[next_opt]
                self.cur_opt = o(**kw)
                # DE needs iteration number to start evolving the initial population without evaluating it
                self.cur_opt.init_with_rh(self.rh, iteration=self._num_iters)

            self.cur_opt_str = next_opt

        print("TempoRL selection done!")
        try:
            # to safeguard try failing in the first step
            # to ensure self._x_guess and x_guess have been initialized
            # self._x_guess should allow tracking of configurations such that after a switch
            #  to a working optimizer, the evaluations can still be used in expected format
            if self.cur_opt_str == "Warmstart":
                x_guess = self.original_cs.sample_configuration(n_suggestions)
                if n_suggestions == 1:
                    x_guess = [x_guess]
                x_guess_to_keep = []
                for guess in x_guess:
                    transformed_guess = dict()
                    for hp_name in guess:
                        if hp_name in self._par:
                            hp_value = TRANS[self._par[hp_name]](guess[hp_name])
                        else:
                            hp_value = guess[hp_name]
                        transformed_guess[hp_name] = hp_value
                    x_guess_to_keep.append(Configuration(self.cs, values=transformed_guess))
                self._x_guess = x_guess_to_keep
            else:
                x_guess = self.cs.sample_configuration(n_suggestions)
                self._x_guess = copy.deepcopy(x_guess)
                for i, x in enumerate(x_guess):
                    try:
                        x_guess[i] = self._invert_bilog_logit(x)
                    except Exception as e:  # if inversion fails
                        print("Exception during bilog inversion at {}/{}: {}".format(i+1, len(x_guess), e))
                        # sampling from the original space
                        x_guess[i] = self.original_cs.sample_configuration(1)
                        # routine to copy x_guess[i] to self._x_guess[i], while handling
                        # the types and transforms associated
                        import pdb
                        pdb.set_trace()
                        for pname in self._api_config.keys():
                            if pname in self._par:
                                pspace = self._api_config[pname]['space']
                                transformed_value = TRANS[pspace](x_guess[i][pname])
                                self._x_guess[i][pname] = transformed_value
                            else:
                                self._x_guess[i][pname] = x_guess[i][pname]

            #################################
            # beginning of actual algorithm #
            #################################
            x_guess = self.cur_opt.suggest(n_suggestions=n_suggestions)
            if self.cur_opt_str == "Warmstart":
                # the x_guess we keep needs bilog/logit transform
                x_guess_to_keep = []
                for guess in x_guess:
                    transformed_guess = dict()
                    for hp_name in guess:
                        if hp_name in self._par:
                            hp_value = TRANS[self._par[hp_name]](guess[hp_name])
                        else:
                            hp_value = guess[hp_name]
                        transformed_guess[hp_name] = hp_value
                    x_guess_to_keep.append(Configuration(self.cs, values=transformed_guess))
                self._x_guess = x_guess_to_keep
            else:
                # NOTE: `self._x_guess` corresponds to `self.cs`
                # `x_guess` corresponds to `self.original_cs`
                self._x_guess = copy.deepcopy(x_guess)
                for i, x in enumerate(x_guess):
                    try:
                        x_guess[i] = self._invert_bilog_logit(x)
                    except Exception as e:  # if inversion fails
                        print("Exception during bilog inversion at {}/{}: {}".format(i+1, len(x_guess), e))
                        # sampling from the original space
                        x_guess[i] = self.original_cs.sample_configuration(1)
                        # routine to copy x_guess[i] to self._x_guess[i], while handling
                        # the types and transforms associated
                        for pname in self._api_config.keys():
                            if pname in self._par:
                                pspace = self._api_config[pname]['space']
                                transformed_value = TRANS[pspace](x_guess[i][pname])
                                self._x_guess[i][pname] = transformed_value
                            else:
                                self._x_guess[i][pname] = x_guess[i][pname]
        except Exception as e:
            print(e)
        x_guess = [guess.get_dictionary() for guess in x_guess]

        print("Iteration: %d, Current Optimizer: %s" % ((self._num_iters + 1), self.cur_opt_str))
        self._num_iters += 1
        print("\nENDING SUGGEST(): ")

        return x_guess

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
        print("\nBEGINNING OBSERVE(): ")
        # to handle NaN/inf --- detect and replace with maximum y observed so far
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            replace_ids = np.isnan(y) + np.isinf(y)

            # if all the observed y are not NaN/inf, use the max of the observed legitimate y's
            if not np.all(replace_ids):
                max_y = np.array(y)[(~replace_ids).nonzero()[0]].max()
            else:
                # avoiding random sampling since different metrics will have different ranges/sign
                # for the case where some legitimate run history was recorded, replacing with a
                #   random sample from a fixed range can provide incorrect signals
                max_y = np.finfo(np.float32).max

            # replace max_y with the max value of y in the non-empty recorded history
            if len(self.rh) > 1:
                res = pd.DataFrame(self.rh).iloc[:, -1]  # collecting y values
                # if the current observation contains a legitimate y-value and the recorded
                #  history contains the highest float value -> prefer max of legitimate observation
                _max_y = pd.DataFrame(res).iloc[:, -1].max()
                # this if...else should fetch the maximum y that is not the maximal float
                if max_y < np.finfo(float).max and _max_y == np.finfo(float).max:
                    pass  # max_y remains unchanged
                else:
                    # max of all recorded observations including current iteration
                    max_y = np.max((_max_y, max_y))

            # replacing the NaN/inf with the chosen value
            for idx in replace_ids.nonzero()[0].tolist():
                y[idx] = max_y

        # beginning of actual algorithm
        for c, res in zip(self._x_guess, y):
            self.rh.append([c, res])
        self.cur_opt.observe(self._x_guess, y)
        print("\nENDING OBSERVE(): ")

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

    def _invert_bilog_logit(self, x):
        dictionary = copy.copy(x) if isinstance(x, dict) else x.get_dictionary()
        for k, v in dictionary.items():
            if k in self._par:
                hp = self.original_cs.get_hyperparameter(k)
                _fun = INV_TRANS[self._par[k]]
                dictionary[k] = np.clip(_fun(v), hp.lower, hp.upper)
                # need to check  original configspace since bilog-int are converted to float
                if isinstance(self.original_cs.get_hyperparameter(k), UniformIntegerHyperparameter):
                    dictionary[k] = int(np.rint(dictionary[k]))
        x = Configuration(self.original_cs, values=dictionary)
        return x


if __name__ == "__main__":
    experiment_main(SwitchingOptimizer)

    # The test for the bilog spaces.. Please uncomment for testing
    # api_config = {'hidden_layer_sizes': {'type': 'int', 'space': 'linear', 'range': (50, 200)},
    #  'learning_rate_init': {'type': 'real', 'space': 'bilog', 'range': (-1, 1)},  # To test negative value for `bilog`
    #  'beta_1': {'type': 'real', 'space': 'logit', 'range': (0.5, 0.99)},
    #  'epsilon': {'type': 'real', 'space': 'logit', 'range': (1e-9, 1e-6)}
    # }
    #
    # opt = SwitchingOptimizer(api_config)
    #
    # for i in range(20):
    #     X = opt.suggest(n_suggestions=8)
    #     opt.observe(X, list(range(8)))

