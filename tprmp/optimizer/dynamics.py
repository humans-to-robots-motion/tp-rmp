import cvxpy as cp
import numpy as np
import logging

from tprmp.models.rmp import compute_obsrv_prob, compute_policy, compute_potential_term, compute_riemannian_metric

logger = logging.getLogger(__name__)


def optimize_dynamics(tp_hsmm, demos, **kwargs):
    alpha = kwargs.get('alpha', 1e-5)
    beta = kwargs.get('beta', 1e-5)
    stiff_scale = kwargs.get('stiff_scale', 1.)
    tau = kwargs.get('tau', 1.)
    potential_method = kwargs.get('potential_method', 'quadratic')
    d_min = kwargs.get('d_min', 0.)
    energy = kwargs.get('energy', 0.)
    verbose = kwargs.get('verbose', False)
    phi0_frames = optimize_potentials(tp_hsmm, demos, alpha=alpha, stiff_scale=stiff_scale, tau=tau, potential_method=potential_method, energy=energy, verbose=verbose)
    d0_frames = optimize_dissipation(tp_hsmm, demos, phi0_frames, beta=beta, stiff_scale=stiff_scale, tau=tau, potential_method=potential_method, d_min=d_min, verbose=verbose)
    return phi0_frames, d0_frames


def optimize_potentials(tp_hsmm, demos, **kwargs):
    alpha = kwargs.get('alpha', 1e-5)
    stiff_scale = kwargs.get('stiff_scale', 1.)
    tau = kwargs.get('tau', 1.)
    potential_method = kwargs.get('potential_method', 'quadratic')
    energy = kwargs.get('energy', 0.)
    eps = kwargs.get('eps', 1e-4)
    verbose = kwargs.get('verbose', False)
    frames = demos[0].frame_names
    gap = energy / tp_hsmm.num_comp
    phi0_frames = {}
    for frame in frames:
        phi0 = cp.Variable(tp_hsmm.num_comp)
        loss = 0.
        for demo in demos:
            trajs = demo.traj_in_frames[frame]
            x, dx = trajs['traj'], trajs['d_traj']
            for t in range(x.shape[1]):
                mvns = tp_hsmm.get_local_gmm(frame)
                weights = compute_obsrv_prob(x[:, t], mvns)
                f = compute_potential_term(weights, phi0, x[:, t], mvns, stiff_scale=stiff_scale, tau=tau, potential_method=potential_method)
                v = dx[:, t]
                norm_v = np.linalg.norm(v)
                if norm_v > eps:
                    v = v / norm_v
                loss += (cp.norm(v - f) / x.shape[1])
        loss /= len(demos)
        if alpha > 0.:
            loss += alpha * cp.pnorm(phi0, p=2)**2
        objective = cp.Minimize(loss)  # L2 regularization
        problem = cp.Problem(objective, potential_constraints(phi0, gap))
        problem.solve(verbose=verbose)
        logger.info(f'Opimizing potential for frame {frame}...')
        logger.info(f'Status: {problem.status}')
        logger.info(f'Optimal phi0: {phi0.value}')
        phi0_frames[frame] = phi0.value
    return phi0_frames


def optimize_dissipation(tp_hsmm, demos, phi0_frames, **kwargs):
    beta = kwargs.get('beta', 1e-5)
    stiff_scale = kwargs.get('stiff_scale', 1.)
    tau = kwargs.get('tau', 1.)
    potential_method = kwargs.get('potential_method', 'quadratic')
    d_min = kwargs.get('d_min', 0.)
    d_default = kwargs.get('d_default', 50.)
    verbose = kwargs.get('verbose', False)
    frames = demos[0].frame_names
    d0_frames = {}
    for frame in frames:
        d0 = cp.Variable(tp_hsmm.num_comp)
        loss = 0.
        for demo in demos:
            trajs = demo.traj_in_frames[frame]
            x, dx, ddx = trajs['traj'], trajs['d_traj'], trajs['dd_traj']
            for t in range(x.shape[1]):
                mvns = tp_hsmm.get_local_gmm(frame)
                M = compute_riemannian_metric(x[:, t], mvns)
                M_inv = np.linalg.inv(M)
                f = compute_policy(phi0_frames[frame], d0, x[:, t], dx[:, t], mvns, stiff_scale=stiff_scale, tau=tau, potential_method=potential_method)
                loss += cp.norm(ddx[:, t] - M_inv @ f) / x.shape[1]
        loss /= len(demos)
        if beta > 0.:
            loss += beta * cp.pnorm(d0, p=2)**2  # L2 regularization
        objective = cp.Minimize(loss)
        problem = cp.Problem(objective, [d0 >= d_min])
        try:
            problem.solve(verbose=verbose)
            logger.info(f'Opimizing dissipation for frame {frame}...')
            logger.info(f'Status: {problem.status}')
            logger.info(f'Optimal d0: {d0.value}')
            d0_frames[frame] = d0.value
        except cp.error.SolverError:
            logger.warn(f'Opimizing dissipation for frame {frame} failed! Using d_default {d_default}')
            d0_frames[frame] = 50. * np.ones((tp_hsmm.num_comp, tp_hsmm.manifold.dim_T))
    return d0_frames


def potential_constraints(phi0, gap=0.):
    constraints = []
    for k in range(phi0.size - 1):
        constraints.append(phi0[k] >= (phi0[k + 1] + gap))
    constraints.append(phi0 >= 0)
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
    # test retrieval
    # x0, dx0 = np.array([0, 0.5]), np.zeros(2)
    # visualize_rmp(phi0['obj'], d0['obj'], model.get_local_gmm('obj'), x0, dx0, T, dt, limit=10)
