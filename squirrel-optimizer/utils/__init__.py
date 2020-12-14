from warmstart_helper import warmstart_load
from scipy.special import logit
import numpy as np


TRANS = {
    'logit': logit,
    'bilog': lambda x: np.sign(x) * np.log(1 + np.abs(x))
}

INV_TRANS = {
    'logit': lambda x: 1 / (1 + np.exp(-x)),
    'bilog': lambda x: np.sign(x) * (np.exp(np.abs(x)) - 1)
}
