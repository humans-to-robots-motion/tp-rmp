import numpy as np
import cvxpy as cp

def compute_riemannian_metric(x, mvns):
    weights = compute_obsrv_prob(x, mvns)
    Ms = np.array([comp.cov_inv for comp in mvns])
    return Ms.T @ weights 


def compute_policy(phi0, d0, x, dx, mvns, use_cp=False):
    weights = compute_obsrv_prob(x, mvns)
    return compute_potential_term(weights, phi0, x, mvns) + compute_dissipation_term(weights, d0, dx, mvns, use_cp=use_cp)


def compute_potential_term(weights, phi0, x, mvns):
    phi = compute_potentials(phi0, x, mvns)
    Phi = weights.T @ phi if len(x.shape) == 1 else np.diag(weights.T @ phi)
    num_comp = len(mvns)
    manifold = mvns[0].manifold
    Ps = np.zeros(manifold.dim_T) if len(x.shape) == 1 else np.zeros((manifold.dim_T, x.shape[1]))
    for k in range(num_comp):
        Ps += weights[k] * (phi[k] - Phi) * (mvns[k].cov_inv @ manifold.log_map(x, base=mvns[k].mean))
        Ps += -weights[k] * (mvns[k].cov_inv @ manifold.log_map(x, base=mvns[k].mean))
    return Ps


def compute_dissipation_term(weights, d0, dx, mvns, use_cp=False):
    manifold = mvns[0].manifold
    Ds = np.zeros(manifold.dim_T) if len(dx.shape) == 1 else np.zeros((manifold.dim_T, dx.shape[1]))
    for k in range(len(mvns)):
        if use_cp:
            Ds += -weights[k] * cp.multiply(d0[k], dx)
        else:
            if len(dx.shape) > 1:
                d = np.expand_dims(d0[k], axis=-1)
            else:
                d = d0[k]
            Ds += -weights[k] * d * dx
    return Ds


def compute_potentials(phi0, x, mvns):
    num_comp = len(mvns)
    P = np.zeros(num_comp) if len(x.shape) == 1 else np.zeros((num_comp, x.shape[1]))
    manifold = mvns[0].manifold
    for k in range(num_comp):
        comp = mvns[k]
        v = manifold.log_map(x, base=comp.mean)
        P[k] = (v * (comp.cov_inv @ v)).sum(0)
    phi = phi0 + P if len(x.shape) == 1 else np.expand_dims(phi0, axis=1) + P
    return phi


def compute_obsrv_prob(x, mvns, normalized=True, eps=1e-30):
    num_comp = len(mvns)
    prob = np.zeros(num_comp) if len(x.shape) == 1 else np.zeros((num_comp, x.shape[1]))
    for k in range(num_comp):
        prob[k] = mvns[k].pdf(x)
    if normalized:
        s = prob.sum()
        if s < eps:  # collapse into uniform distribution
            prob = np.ones(num_comp) / num_comp
        else:
            prob /= prob.sum()
    return prob


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from tprmp.demonstrations.probability import ManifoldGaussian
    from tprmp.demonstrations.manifold import Manifold
    from tprmp.visualization.dynamics import visualize_rmp
    manifold = Manifold.get_euclidean_manifold(2)
    mvns = [ManifoldGaussian(manifold, np.ones(2), np.eye(2)),
            ManifoldGaussian(manifold, 3 * np.ones(2), 2 * np.eye(2)),
            ManifoldGaussian(manifold, 7 * np.ones(2), 3 * np.eye(2))]
    T = 300
    dt = 0.05
    phi0 = [5., 1., 0.]
    d0 = .2 * np.ones((3, 2))
    # test semantics
    x, dx = np.zeros((2, T)), np.zeros((2, T))
    print(compute_policy(phi0, d0, x, dx, mvns).shape)
    # test riemannian
    x = np.zeros((2, T))
    print(compute_riemannian_metric(x, mvns).shape)
    # # test dynamics
    x, dx = np.zeros(2), np.zeros(2)
    visualize_rmp(phi0, d0, mvns, x, dx, T, dt, limit=10)
