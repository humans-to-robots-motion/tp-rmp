import numpy as np

from tprmp.models.rmp import compute_obsrv_prob


def compute_coriolis_force(x, dx, mvns, mass_scale=1.):
    weights = compute_obsrv_prob(x, mvns)
    scale = compute_scale(weights, x, mvns)
    return compute_dMdt(weights, scale, dx, mvns, mass_scale=mass_scale) @ dx - 0.5 * compute_dTdx(weights, scale, dx, mvns, mass_scale=mass_scale)


def compute_dMdt(weights, scale, dx, mvns, mass_scale=1.):
    scale = scale @ dx
    Ms = np.array([(mass_scale**2) * comp.cov_inv for comp in mvns])
    dMdt = Ms.T @ (weights * scale)
    return dMdt


def compute_dTdx(weights, scale, dx, mvns, mass_scale=1.):
    dTdx = np.zeros_like(dx)
    for k in range(len(mvns)):
        dTdx += weights[k] * (dx.T @ ((mass_scale**2) * mvns[k].cov_inv) @ dx) * scale[k]
    return dTdx


def compute_scale(weights, x, mvns):
    manifold = mvns[0].manifold
    pulls = np.zeros((len(mvns), manifold.dim_T))
    for k in range(weights.shape[0]):
        pulls[k] = manifold.log_map(x, base=mvns[k].mean).T @ mvns[k].cov_inv
    mean_pull = weights.T @ pulls
    scale = mean_pull - pulls
    return scale


if __name__ == '__main__':
    from tprmp.demonstrations.probability import ManifoldGaussian
    from tprmp.demonstrations.manifold import Manifold
    manifold = Manifold.get_euclidean_manifold(2)
    mvns = [ManifoldGaussian(manifold, np.ones(2), np.eye(2)),
            ManifoldGaussian(manifold, 3 * np.ones(2), np.eye(2)),
            ManifoldGaussian(manifold, 5 * np.ones(2), np.eye(2))]
    # test coriolis forces, should be zeros for all cases
    x, dx = np.ones(2), np.zeros(2)
    print(compute_coriolis_force(x, dx, mvns))
    x, dx = np.ones(2), np.ones(2)
    print(compute_coriolis_force(x, dx, mvns))
    x, dx = 100000 * np.ones(2), 100000 * np.ones(2)
    print(compute_coriolis_force(x, dx, mvns))
