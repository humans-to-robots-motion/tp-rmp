import numpy as np

'''
"Programming by Demonstration on Riemannian Manifolds", M.J.A. Zeestraten, 2018.
'''


def e_log_map(p, base=None):
    if base is None:
        return p
    elif len(p.shape) == 2:
        return p - np.tile(base, (p.shape[1], 1)).T
    else:
        return p - base


def e_exp_map(p, base=None):
    if base is None:
        return p
    elif len(p.shape) == 2:
        return p + np.tile(base, (p.shape[1], 1)).T
    else:
        return p + base


def e_parallel_transport(p, g, h):
    return p
