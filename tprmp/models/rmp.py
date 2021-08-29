import numpy as np


def compute_riemannian_metric(x, mvns, mass_scale=1., eps=1e-5):
    weights = compute_obsrv_prob(x, mvns)
    Ms = np.array([(mass_scale**2) * comp.cov_inv for comp in mvns])
    M = Ms.T @ weights if weights.sum() > 0. else np.eye(mvns[0].manifold.dim_T)
    return M + eps * np.eye(mvns[0].manifold.dim_T)


def compute_hamiltonian(phi0, x, dx, mvns, **kwargs):
    manifold = kwargs.get('manifold', None)
    mass_scale = kwargs.get('mass_scale', 1.)
    weights = compute_obsrv_prob(x, mvns, manifold=manifold)
    phi = compute_potentials(phi0, x, mvns, **kwargs)
    M = compute_riemannian_metric(x, mvns, mass_scale=mass_scale)
    T = 0.5 * dx.T @ M @ dx
    Phi = weights.T @ phi
    return T + Phi


def compute_policy(phi0, d0, x, dx, mvns, **kwargs):
    weights = compute_obsrv_prob(x, mvns)
    return compute_potential_term(weights, phi0, x, mvns, **kwargs) + compute_dissipation_term(weights, d0, dx)


def compute_potential_term(weights, phi0, x, mvns, **kwargs):
    stiff_scale = kwargs.get('stiff_scale',  1.)
    tau = kwargs.get('tau',  1.)
    delta = kwargs.get('delta',  0.1)
    potential_method = kwargs.get('potential_method',  'quadratic')
    manifold = kwargs.get('manifold', None)
    phi = compute_potentials(phi0, x, mvns, **kwargs)
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
            norm = np.sqrt((stiff_scale**2) * v.T @ pulls[k])
            Ps += -weights[k] * np.tanh(tau * norm) * (stiff_scale**2) * pulls[k] / norm
        elif potential_method == 'huber':  # huber potential does not use mvns variance shape
            v = manifold.log_map(x, base=mvns[k].mean)
            quadratic = (stiff_scale**2) * v.T @ np.eye(manifold.dim_T) @ v
            norm = np.sqrt(quadratic)
            if norm <= delta:
                Ps += -weights[k] * (stiff_scale**2) * v
            else:
                Ps += -weights[k] * (stiff_scale**2) * delta * v / norm
        else:
            raise ValueError(f'Potential method {potential_method} is unrecognized!')
    return Ps


def compute_dissipation_term(weights, d0, dx):
    Ds = -(weights @ d0) * dx
    return Ds


def compute_potentials(phi0, x, mvns, **kwargs):
    stiff_scale = kwargs.get('stiff_scale',  1.)
    tau = kwargs.get('tau',  1.)
    delta = kwargs.get('delta',  0.1)
    potential_method = kwargs.get('potential_method',  'quadratic')
    manifold = kwargs.get('manifold', None)
    num_comp = len(mvns)
    P = np.zeros(num_comp)
    if manifold is None:
        manifold = mvns[0].manifold
    d = manifold.dim_M
    for k in range(num_comp):
        comp = mvns[k]
        v = manifold.log_map(x[:d], base=comp.mean[:d])
        quadratic = (stiff_scale**2) * v.T @ comp.cov_inv[:d, :d] @ v
        if potential_method == 'quadratic':
            P[k] = 0.5 * quadratic
        elif potential_method == 'tanh':
            norm = np.sqrt(quadratic)
            P[k] = 1 / tau * (np.exp(tau * norm) + np.exp(-tau * norm))
        elif potential_method == 'huber':  # huber potential does not use mvns variance shape
            quadratic = (stiff_scale**2) * v.T @ np.eye(manifold.dim_T) @ v
            norm = np.sqrt(quadratic)
            if norm <= delta:
                P[k] = 0.5 * quadratic
            else:
                P[k] = delta * (norm - 0.5 * delta)
        else:
            raise ValueError(f'Potential method {potential_method} is unrecognized!')
    phi = phi0 + P
    return phi


def compute_obsrv_prob(x, mvns, normalized=True, eps=1e-307, manifold=None):
    num_comp = len(mvns)
    prob = np.zeros(num_comp) if len(x.shape) == 1 else np.zeros((num_comp, x.shape[1]))
    for k in range(num_comp):
        prob[k] = mvns[k].pdf(x, manifold=manifold)
    if normalized:
        s = prob.sum()
        if s > eps:
            prob /= s
    return prob
