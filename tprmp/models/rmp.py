import numpy as np


def compute_riemannian_metric(x, mvns, eps=1e-3):
    weights = compute_obsrv_prob(x, mvns)
    Ms = np.array([comp.cov_inv for comp in mvns])
    M = Ms.T @ weights if weights.sum() == 1. else np.eye(mvns[0].manifold.dim_T)
    return M + eps * np.eye(mvns[0].manifold.dim_T)


def compute_policy(phi0, d0, x, dx, mvns, stiff_scale=1., tau=1., potential_method='quadratic'):
    weights = compute_obsrv_prob(x, mvns)
    return compute_potential_term(weights, phi0, x, mvns, stiff_scale=stiff_scale, tau=tau, potential_method=potential_method) + compute_dissipation_term(weights, d0, dx, mvns)


def compute_potential_term(weights, phi0, x, mvns, stiff_scale=1., tau=1., potential_method='quadratic'):
    '''stiff_scale only works with quadratic potential'''
    phi = compute_potentials(phi0, x, mvns, stiff_scale=stiff_scale, tau=tau, potential_method=potential_method)
    num_comp = len(mvns)
    manifold = mvns[0].manifold
    Ps = np.zeros(manifold.dim_T)
    pulls = np.zeros((num_comp, manifold.dim_T))
    for k in range(num_comp):
        pulls[k] = mvns[k].cov_inv @ manifold.log_map(x, base=mvns[k].mean)
    mean_pull = weights.T @ pulls
    for k in range(num_comp):
        Ps += weights[k] * phi[k] * (pulls[k] - mean_pull)
        if potential_method == 'quadratic':
            Ps += -weights[k] * (stiff_scale**2) * pulls[k]
        elif potential_method == 'tanh':
            v = manifold.log_map(x, base=mvns[k].mean)
            norm = np.sqrt(v.T @ pulls[k])
            Ps += -weights[k] * np.tanh(tau * norm) * pulls[k] / norm
        else:
            raise ValueError(f'Potential method {potential_method} is unrecognized!')
    return Ps


def compute_dissipation_term(weights, d0, dx, mvns):
    manifold = mvns[0].manifold
    Ds = np.zeros(manifold.dim_T)
    for k in range(len(mvns)):
        Ds += -weights[k] * d0[k] * dx
    return Ds


def compute_potentials(phi0, x, mvns, stiff_scale=1., tau=1., potential_method='quadratic'):
    num_comp = len(mvns)
    P = np.zeros(num_comp)
    manifold = mvns[0].manifold
    for k in range(num_comp):
        comp = mvns[k]
        v = manifold.log_map(x, base=comp.mean)
        quadratic = v.T @ ((stiff_scale**2) * comp.cov_inv) @ v
        if potential_method == 'quadratic':
            P[k] = quadratic
        elif potential_method == 'tanh':
            norm = np.sqrt(quadratic)
            P[k] = 1 / tau * (np.exp(tau * norm) + np.exp(-tau * norm))
        else:
            raise ValueError(f'Potential method {potential_method} is unrecognized!')
    phi = phi0 + P
    return phi


def compute_obsrv_prob(x, mvns, normalized=True, eps=1e-307):
    num_comp = len(mvns)
    prob = np.zeros(num_comp) if len(x.shape) == 1 else np.zeros((num_comp, x.shape[1]))
    for k in range(num_comp):
        prob[k] = mvns[k].pdf(x)
    if normalized:
        s = prob.sum()
        if s > eps:
            prob /= s
    return prob
