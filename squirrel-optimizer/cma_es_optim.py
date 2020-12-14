import numpy as np
import logging

from copy import copy
from typing import Callable, Any, Tuple, List, Union, Dict

from GaussianProcess import GaussianProcess
from GaussianProcess.trend import constant_trend
from cma import CMAEvolutionStrategy
from cma.optimization_tools import BestSolution

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter

Vector = List[float]
Matrix = List[Vector]


def handle_box_constraint(x, lb, ub):
    """This function transforms x to t w.r.t. the low and high
    boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
    described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
    for Parameter Optimization and Their Applications to Medical Image
    Analysis" as alorithm 6.
    """
    x = np.asarray(x, dtype='float')
    shape_ori = x.shape
    x = np.atleast_2d(x)
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)

    transpose = False
    if x.shape[0] != len(lb):
        x = x.T
        transpose = True

    lb, ub = lb.flatten(), ub.flatten()
    lb_index = np.isfinite(lb)
    up_index = np.isfinite(ub)

    valid = np.bitwise_and(lb_index, up_index)

    LB = lb[valid][:, np.newaxis]
    UB = ub[valid][:, np.newaxis]

    y = (x[valid, :] - LB) / (UB - LB)
    I = np.mod(np.floor(y), 2) == 0
    yprime = np.zeros(y.shape)
    yprime[I] = np.abs(y[I] - np.floor(y[I]))
    yprime[~I] = 1.0 - np.abs(y[~I] - np.floor(y[~I]))

    x[valid, :] = LB + (UB - LB) * yprime

    if transpose:
        x = x.T
    return x.reshape(shape_ori)


def vector_to_configspace(cs, vector):
    '''Converts numpy array to ConfigSpace object
    Works when self.cs is a ConfigSpace object and each component of the
    input vector is in the domain [0, 1].
    '''
    new_config = cs.sample_configuration()
    for i, hyper in enumerate(cs.get_hyperparameters()):
        if type(hyper) == OrdinalHyperparameter:
            ranges = np.arange(start=0, stop=1, step=1 / len(hyper.sequence))
            param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
        elif type(hyper) == CategoricalHyperparameter:
            ranges = np.arange(start=0, stop=1, step=1 / len(hyper.choices))
            param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
        else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
            # rescaling continuous values
            param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
            if type(hyper) == UniformIntegerHyperparameter:
                param_value = np.round(param_value).astype(int)  # converting to discrete (int)
        new_config[hyper.name] = param_value
    return new_config


class CMA(CMAEvolutionStrategy):
    def __init__(
            self,
            cs,
            popsize: int,
            lb: Union[float, str, Vector, np.ndarray] = -np.inf,
            ub: Union[float, str, Vector, np.ndarray] = np.inf,
            ftarget: Union[int, float] = -np.inf,
            max_FEs: Union[int, str] = np.inf,
            verbose: bool = False,
            logger=None
        ):
        """Wrapper Class for `pycma`

        Parameters
        ----------
        dim : int
            dimensionality
        popsize : int
            population size
        lb : Union[float, str, Vector, np.ndarray], optional
            lower bound of the decision space, by default -np.inf
        ub : Union[float, str, Vector, np.ndarray], optional
            upper bound of the decision space, by default np.inf
        ftarget : Union[int, float], optional
            the target value, by default -np.inf
        max_FEs : Union[int, str], optional
            the evaluation budget, by default np.inf
        verbose : bool, optional
            the verbosity, by default False
        logger : optional
            a logger object, by default None
        """

        inopts = {
            'bounds': [lb, ub],
            'ftarget': ftarget,
            'popsize': popsize
        }
        sigma0 = (ub - lb) / 5
        dim = len(cs.get_hyperparameters())
        ub = np.array([ub] * dim)
        lb = np.array([lb] * dim)
        x0 = (ub - lb) * np.random.rand(dim) + lb

        super().__init__(x0=x0, sigma0=sigma0, inopts=inopts)
        self.dim = dim
        self.logger = logger
        self.max_FEs = max_FEs
        self.ftarget = ftarget
        self.verbose = verbose
        self.stop_dict = {}
        self.cs = cs

    def init_with_rh(self, data, **kwargs):
        X = np.atleast_2d([
            Configuration(values=_[0], configuration_space=self.cs).get_array()\
                 for _ in data
        ])
        y = np.array([_[1] for _ in data])
        dim = X.shape[1]
        fopt = np.min(y)
        xopt = X[np.where(y == fopt)[0][0]]

        mean = constant_trend(dim, beta=None)  # Simple Kriging
        thetaL = 1e-10 * np.ones(dim)
        thetaU = 10 * np.ones(dim)
        theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

        model = GaussianProcess(
            mean=mean, corr='squared_exponential',
            theta0=theta0, thetaL=thetaL, thetaU=thetaU,
            nugget=1e-6, noise_estim=False,
            optimizer='BFGS', wait_iter=5, random_start=5 * dim,
            eval_budget=100 * dim
        )
        model.fit(X, y)

        # obtain the Hessian and gradient from the GP mean surface
        H = model.Hessian(xopt)
        g = model.gradient(xopt)[0]

        w, B = np.linalg.eigh(H)
        w[w <= 0] = 1e-6     # replace the negative eigenvalues by a very small value
        w_min, w_max = np.min(w), np.max(w)

        # to avoid the conditional number gets too high
        cond_upper = 1e3
        delta = (cond_upper * w_min - w_max) / (1 - cond_upper)
        w += delta

        # compute the upper bound for step-size
        M = np.diag(1 / np.sqrt(w)).dot(B.T)
        H_inv = B.dot(np.diag(1 / w)).dot(B.T)
        p = -1 * H_inv.dot(g).ravel()
        alpha = np.linalg.norm(p)

        if np.isnan(alpha):
            alpha = 1
            H_inv = np.eye(dim)

        # use a backtracking line search to determine the initial step-size
        tau, c = 0.9, 1e-4
        slope = np.inner(g.ravel(), p.ravel())

        if slope > 0:  # this should not happen..
            p *= -1
            slope *= -1

        f = lambda x: model.predict(x)
        while True:
            _x = (xopt + alpha * p).reshape(1, -1)
            if f(_x) <= f(xopt.reshape(1, -1)) + c * alpha * slope:
                break
            alpha *= tau

        sigma0 = np.linalg.norm(M.dot(alpha * p)) / np.sqrt(dim - 0.5)
        self.Cov = H_inv
        self.sigma = self.sigma0 = sigma0
        self._set_x0(xopt)
        self.mean = self.gp.geno(
            np.array(self.x0, copy=True),
            from_bounds=self.boundary_handler.inverse,
            copy=False
        )
        self.mean0 = np.array(self.mean, copy=True)
        self.best = BestSolution(x=self.mean, f=fopt)

    @property
    def eval_count(self):
        return self.countevals

    @property
    def iter_count(self):
        return self.countiter

    @property
    def x(self):
        return self.mean

    @x.setter
    def x(self, x):
        self.mean = copy(x)

    @property
    def Cov(self):
        return self.C

    @Cov.setter
    def Cov(self, C):
        try:
            w, B = np.linalg.eigh(C)
            if np.all(np.isreal(w)):
                self.B = self.sm.B = B
                self.D = self.sm.D = w ** 0.5
                self.dC = np.diag(C)
                self.C = self.sm.C = C
                self.sm._sortBD()
        except np.linalg.LinAlgError:
            pass

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        if isinstance(logger, logging.Logger):
            self._logger = logger
            self._logger.propagate = False
            return

    def suggest(self, n_suggestions: int = 1) -> List[Dict]:
        try:
            _X = super().ask(number=n_suggestions)
            self._X = [handle_box_constraint(x, 0, 1) for x in _X]
        except Exception as e:
            print(e)
        return [vector_to_configspace(self.cs, x) for x in self._X]

    def observe(self, X, y):
        super().tell(self._X, y)

    def check_stop(self):
        _, f, __ = self.best.get()
        if f <= self.ftarget:
            self.stop_dict['ftarget'] = f

        if self.countevals >= self.max_FEs:
            self.stop_dict['FEs'] = self.countevals

        return bool(self.stop_dict)