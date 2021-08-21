import numpy as np

from tprmp.models.rmp import compute_obsrv_prob


def compute_coriolis_force(x, dx, mvns):
    weights = compute_obsrv_prob(x, mvns)
    return compute_dMdt_term(weights, x, dx, mvns) - compute_dTdx_term(weights, x, dx, mvns)


def compute_dMdt_term(weights, x, dx, mvns):
    terms = np.zeros_like(weights)
    manifold = mvns[0].manifold
    for k in range(weights.shape[0]):
        terms[k] = manifold.log_map(x, base=mvns[k].mean).T @ mvns[k].cov_inv @ dx
    weighted_term = weights.T @ terms
    scale = weighted_term - terms
    Ms = np.array([comp.cov_inv for comp in mvns])
    dMdt = Ms.T @ (weights * scale)
    return dMdt @ dx


def compute_dTdx_term(weights, x, dx, mvns):
    manifold = mvns[0].manifold
    dim_T = manifold.dim_T
    dTdx = np.zeros(dim_T)
    terms = np.zeros((weights.shape[0], dim_T))
    Ms = np.array([comp.cov_inv for comp in mvns])
    for i in range(dim_T):
        for k in range(weights.shape[0]):
            terms[k, i] = manifold.log_map(x, base=mvns[k].mean).T @ mvns[k].cov_inv[i, :]
        weighted_term = weights.T @ terms[:, i]
        scale = weighted_term - terms[:, i]
        dMdxi = Ms.T @ (weights * scale)
        dTdx[i] = dx.T @ dMdxi @ dx
    return 1. / 2. * dTdx


if __name__ == '__main__':
    from tprmp.demonstrations.probability import ManifoldGaussian
    from tprmp.demonstrations.manifold import Manifold
    manifold = Manifold.get_euclidean_manifold(2)
    mvns = [ManifoldGaussian(manifold, np.ones(2), np.eye(2)),
            ManifoldGaussian(manifold, 3 * np.ones(2), 2 * np.eye(2)),
            ManifoldGaussian(manifold, 7 * np.ones(2), 3 * np.eye(2))]
    x, dx = np.ones(2), np.zeros(2)
    print(compute_coriolis_force(x, dx, mvns))
    x, dx = np.ones(2), np.ones(2)
    print(compute_coriolis_force(x, dx, mvns))
    x, dx = 2 * np.ones(2), np.ones(2)
    print(compute_coriolis_force(x, dx, mvns))
