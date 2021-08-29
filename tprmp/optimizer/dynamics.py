import cvxpy as cp
import numpy as np
import logging

from tprmp.models.coriolis import compute_coriolis_force
from tprmp.models.rmp import compute_dissipation_term, compute_hamiltonian, compute_obsrv_prob, compute_policy, compute_potential_term, compute_riemannian_metric

logger = logging.getLogger(__name__)


def optimize_dynamics(tp_gmm, demos, **kwargs):
    alpha = kwargs.get('alpha', 1e-5)
    beta = kwargs.get('beta', 1e-5)
    stiff_scale = kwargs.get('stiff_scale', 1.)
    mass_scale = kwargs.get('mass_scale', 1.)
    tau = kwargs.get('tau', 1.)
    delta = kwargs.get('delta', 1.)
    potential_method = kwargs.get('potential_method', 'quadratic')
    train_method = kwargs.get('train_method', 'match_accel')
    d_min = kwargs.get('d_min', 0.)
    energy = kwargs.get('energy', 0.)
    verbose = kwargs.get('verbose', False)
    phi0 = optimize_potentials(tp_gmm, demos, alpha=alpha, stiff_scale=stiff_scale, tau=tau, delta=delta, potential_method=potential_method, energy=energy, verbose=verbose)
    d0 = optimize_dissipation(tp_gmm, demos, phi0, beta=beta, stiff_scale=stiff_scale, mass_scale=mass_scale,
                              tau=tau, delta=delta, potential_method=potential_method, train_method=train_method, d_min=d_min, verbose=verbose)
    return phi0, d0


def optimize_potentials(tp_gmm, demos, **kwargs):
    alpha = kwargs.get('alpha', 1e-5)
    stiff_scale = kwargs.get('stiff_scale', 1.)
    tau = kwargs.get('tau', 1.)
    delta = kwargs.get('delta', 1.)
    potential_method = kwargs.get('potential_method', 'quadratic')
    energy = kwargs.get('energy', 0.)
    eps = kwargs.get('eps', 1e-4)
    verbose = kwargs.get('verbose', False)
    gap = energy / tp_gmm.num_comp
    phi0 = cp.Variable(tp_gmm.num_comp)
    loss = 0.
    for demo in demos:
        x, dx = demo.traj, demo.d_traj
        mvns = tp_gmm.generate_global_gmm(demo.get_task_parameters())
        for t in range(x.shape[1]):
            weights = compute_obsrv_prob(x[:, t], mvns)
            f = compute_potential_term(weights, phi0, x[:, t], mvns, stiff_scale=stiff_scale, tau=tau, delta=delta, potential_method=potential_method)
            v = dx[:, t]
            norm_v = np.linalg.norm(v)
            if norm_v > eps:
                v = v / norm_v
            loss += (cp.norm(v - f) / x.shape[1])
    loss /= len(demos)
    if alpha > 0.:
        loss += alpha * cp.pnorm(phi0, p=2)**2  # L2 regularization
    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, potential_constraints(phi0, gap))
    problem.solve(verbose=verbose)
    logger.info('Optimizing potential...')
    logger.info(f'Status: {problem.status}')
    logger.info(f'Final loss: {loss.value}')
    logger.info(f'Optimal phi0: {phi0.value}')
    return phi0.value


def optimize_dissipation(tp_gmm, demos, phi0, **kwargs):
    beta = kwargs.get('beta', 1e-5)
    stiff_scale = kwargs.get('stiff_scale', 1.)
    mass_scale = kwargs.get('mass_scale', 1.)
    tau = kwargs.get('tau', 1.)
    delta = kwargs.get('delta', 1.)
    potential_method = kwargs.get('potential_method', 'quadratic')
    train_method = kwargs.get('train_method', 'match_accel')
    d_min = kwargs.get('d_min', 0.)
    d_default = kwargs.get('d_default', 50.)
    max_iters = kwargs.get('max_iters', 500)
    verbose = kwargs.get('verbose', False)
    d0 = cp.Variable(tp_gmm.num_comp)
    loss = 0.
    for demo in demos:
        x, dx, ddx = demo.traj, demo.d_traj, demo.dd_traj
        mvns = tp_gmm.generate_global_gmm(demo.get_task_parameters())
        if train_method == 'match_accel':
            for t in range(x.shape[1]):
                M = compute_riemannian_metric(x[:, t], mvns, mass_scale=mass_scale)
                M_inv = np.linalg.inv(M)
                f = compute_policy(phi0, d0, x[:, t], dx[:, t], mvns, stiff_scale=stiff_scale, tau=tau, delta=delta, potential_method=potential_method)
                f -= compute_coriolis_force(x[:, t], dx[:, t], mvns, mass_scale=mass_scale)
                loss += cp.norm(ddx[:, t] - M_inv @ f)
        elif train_method == 'match_energy':
            energy = compute_hamiltonian(phi0, x[:, 0], dx[:, 0], mvns, stiff_scale=stiff_scale, mass_scale=mass_scale, tau=tau, delta=delta, potential_method=potential_method)
            d_energy = 0.  # d_energy is negative
            for t in range(x.shape[1] - 1):
                weights = compute_obsrv_prob(x[:, t], mvns)
                d_energy += compute_dissipation_term(weights, d0, dx[:, t]) @ (dx[:, t] * demo.dt)
            loss += cp.norm(energy + d_energy)
        else:
            raise ValueError(f'Dissipation training method {train_method} is unrecognized!')
    loss /= len(demos)
    if beta > 0.:
        loss += beta * cp.pnorm(d0, p=2)**2  # L2 regularization
    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, dissipation_constraints(d0, d_min))
    try:
        problem.solve(max_iters=max_iters, verbose=verbose)
        logger.info('Optimizing dissipation...')
        logger.info(f'Status: {problem.status}')
        logger.info(f'Final loss: {loss.value}')
        logger.info(f'Optimal d0: {d0.value}')
        res = d0.value
    except cp.error.SolverError:
        logger.warn(f'Optimizing dissipation failed! Using d_default {d_default}')
        res = d_default * np.ones(tp_gmm.num_comp)
    return res


def potential_constraints(phi0, gap=0.):
    constraints = []
    for k in range(phi0.size - 1):
        constraints.append(phi0[k] >= (phi0[k + 1] + gap))
    constraints.append(phi0[phi0.size - 1] >= 0)
    return constraints


def dissipation_constraints(d0, d_min=0.):
    constraints = []
    for k in range(d0.size - 1):
        constraints.append(d0[k] <= d0[k + 1])
    constraints.append(d0[0] >= d_min)
    return constraints


if __name__ == '__main__':
    from tprmp.demonstrations.probability import ManifoldGaussian
    from tprmp.demonstrations.manifold import Manifold
    from tprmp.demonstrations.base import Demonstration
    from tprmp.models.tp_gmm import TPGMM
    # from tprmp.visualization.dynamics import visualize_rmp
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    def simple_demo(T, dt):
        ddx = np.ones((2, T))
        ddx[:, int(T / 2):] = -1
        dx = np.cumsum(ddx, axis=1) * dt
        x = np.cumsum(dx, axis=1) * dt
        return x, dx, ddx

    manifold = Manifold.get_euclidean_manifold(2)
    num_comp = 10
    max_range = 4
    means = np.linspace(1, max_range, num_comp)
    var = (max_range - 1) / (2 * num_comp)
    scale_var = 1.
    mvns = [{'obj': ManifoldGaussian(manifold, means[k] * np.ones(2), scale_var * var * np.eye(2))} for k in range(num_comp)]
    T = 400
    dt = 0.01
    x, dx, ddx = simple_demo(T, dt)
    demo = Demonstration(x, dt=dt)
    demo.add_frame_from_pose(np.zeros(2), 'obj')
    model = TPGMM(num_comp=num_comp)
    model._mvns = mvns
    model._frame_names = ['obj']
    # test training
    phi0, d0 = optimize_dynamics(model, [demo], alpha=1e-5, beta=1e-5)
