import cvxpy as cp
import logging
import sys
sys.path.append('.')

from tprmp.models.rmp import compute_policy

logger = logging.getLogger(__name__)


def optimize_dynamics(tp_hsmm, demos, alpha=0.01, beta=0.5):
    frames = demos[0].frame_names
    phi0_frames = {}
    d0_frames = {}
    for frame in frames:
        phi0 = cp.Variable(tp_hsmm.num_comp)
        d0 = cp.Variable(tp_hsmm.num_comp)
        loss = 0.
        for demo in demos:
            trajs = demo.traj_in_frames[frame]
            x, dx, ddx = trajs['traj'], trajs['d_traj'], trajs['dd_traj']
            for t in range(x.shape[1]):
                loss += cp.sum_squares(ddx[:, t] - compute_policy(phi0, d0, x[:, t], dx[:, t], tp_hsmm.get_local_gmm(frame))) / x.shape[1]
        objective = cp.Minimize(loss / len(demos) + alpha * cp.pnorm(phi0, p=2)**2 + beta * cp.pnorm(d0, p=2)**2)  # L2 regularization
        problem = cp.Problem(objective, field_constraints(phi0, d0))
        problem.solve()
        logger.info(f'Opimizing dynamics for frame {frame}...')
        logger.info(f'Status: {problem.status}')
        logger.info(f'Optimal phi0: {phi0.value}')
        logger.info(f'Optimal d0: {d0.value}')
        phi0_frames[frame] = phi0.value
        d0_frames[frame] = d0.value
    return phi0_frames, d0_frames


def field_constraints(phi0, d0, eps=1e-2):
    constraints = []
    for k in range(phi0.size - 1):
        constraints.append(phi0[k] >= phi0[k + 1])
    constraints.extend([phi0 >= 0, d0 >= eps])
    return constraints


if __name__ == '__main__':
    import numpy as np
    from tprmp.demonstrations.probability import ManifoldGaussian
    from tprmp.demonstrations.manifold import Manifold
    from tprmp.demonstrations.base import Demonstration
    from tprmp.models.tp_gmm import TPGMM
    import matplotlib.pyplot as plt
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    def simple_demo(T, dt):
        ddx = np.ones((2, T))
        ddx[:, int(T / 2):] = -1
        dx = np.cumsum(ddx, axis=1) * dt
        x = np.cumsum(dx, axis=1) * dt
        return x, dx, ddx

    manifold = Manifold.get_euclidean_manifold(2)
    num_comp = 20
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
    phi0, d0 = optimize_dynamics(model, [demo], alpha=0., beta=0.)
    # test retrieval
    x, dx = np.zeros(2), np.zeros(2)
    traj, traj_vel = [x], [dx]
    for t in range(T * 5):
        ddx = compute_policy(phi0['obj'], d0['obj'], x, dx, model.get_local_gmm('obj'))
        dx = ddx * dt + dx
        x = dx * dt + x
        traj.append(x)
        traj_vel.append(dx)
    traj, traj_vel = np.array(traj).T, np.array(traj_vel).T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(traj[0, :], traj[1, :], marker="o", color="b", markersize=1, linestyle='None')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_aspect('equal', 'box')
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(range(T * 5 + 1), traj_vel[0, :], color='r')
    ax = fig.add_subplot(212)
    ax.plot(range(T * 5 + 1), traj_vel[1, :], color='g')
    plt.show()
