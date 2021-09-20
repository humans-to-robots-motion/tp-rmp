import sys
from os.path import join, dirname, abspath
import matplotlib
import logging
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16

ROOT_DIR = join(dirname(abspath(__file__)), '..', '..')
sys.path.append(ROOT_DIR)
from tprmp.utils.loading import load_demos_2d  # noqa
from tprmp.models.tp_rmp import TPRMP  # noqa
from tprmp.models.rmp import compute_riemannian_metric  # noqa
from tprmp.demonstrations.base import Demonstration  # noqa
from tprmp.models.rmp_tree import RMPLeaf, RMPNode, RMPRoot  # noqa
from tprmp.models.rmp_models import CollisionAvoidance  # noqa
from tprmp.visualization.demonstration import plot_frame_2d  # noqa
from tprmp.demonstrations.manifold import Manifold  # noqa
from tprmp.demonstrations.frame import Frame  # noqa

mode = 1
task = 'test'
demo_path = join(ROOT_DIR, 'data', 'tasks', task, 'demos')
num_comp = 9
R = 0.2
moving_goal_radius = 0.5
omega = np.pi
disturb = False
disturb_period = [50, 150]
disturb_magnitude = 10.
N1, N2 = 2, 2
limits = [0., 4.5]
demo_names = ['C1', 'L1', 'P', 'U']
dt = 0.01
v_eps = 8e-2
goal_eps = 0.2
wait = 10
max_steps = 2100
res = 0.05
colormap = 'RdBu' if mode == 1 else 'YlOrBr'
manifold = Manifold.get_euclidean_manifold(2)


def execute(model, root, start_frame, x0, dx0, origin):
    x, dx = x0, dx0
    traj = [x]
    moving = True
    t = 0
    while t < max_steps:
        if moving:
            end_pose = origin + moving_goal_radius * np.array([np.cos(omega * t * dt), np.sin(omega * t * dt)])
            A, b = Demonstration.construct_linear_map(manifold, end_pose)
            end_frame = Frame(A, b, manifold=manifold)
            frames = {'start': start_frame, 'end': end_frame}
        model.generate_global_gmm(frames)
        ddx = root.solve(x, dx)
        if disturb and (t >= disturb_period[0] and t <= disturb_period[1]):
            M = compute_riemannian_metric(x, model._global_mvns, mass_scale=model._mass_scale)
            v = dx / np.linalg.norm(dx)
            df = disturb_magnitude * np.array([v[1], v[0]])
            ddx += np.linalg.inv(M) @ df
        dx = ddx * dt + dx
        x = manifold.exp_map(dx * dt, base=x)
        traj.append(x)
        goal = model._global_mvns[-1].mean
        d = np.linalg.norm(manifold.log_map(x, base=goal))
        if d < goal_eps:
            moving = False
            if np.linalg.norm(dx) < v_eps:
                break
        t += 1
    return np.array(traj).T, frames


X, Y = np.meshgrid(np.arange(limits[0], limits[1], res), np.arange(limits[0], limits[1], res))
fig, axs = plt.subplots(N1, N2)
for n, name in enumerate(demo_names):
    k, m = int(n / N2), n % N2
    data_file = join(demo_path, name + '.p')
    model_file = name + '_' + str(num_comp) + '.p'
    demos = load_demos_2d(data_file, dt=dt)
    model = TPRMP.load(task, model_name=model_file)
    sample = demos[0]
    frames = sample.get_task_parameters()
    x0, dx0 = sample.traj[:, 0], np.zeros(2)
    origin = sample.traj[:, -1]
    # init rmpflow
    T = sample.traj.shape[1]
    cpose = sample.traj[:, int(T / 2)]
    root = RMPRoot('root_space', manifold=manifold)
    ca_node = CollisionAvoidance('CA_space', parent=root, c=cpose, R=R)
    tprmp_node = RMPLeaf('TPRMP_space', model.rmp, parent=root, manifold=manifold, psi=lambda x: x, J=lambda x: np.eye(2))
    traj, frames = execute(model, root, frames['start'], x0, dx0, origin)
    ax = axs[k, m]
    plt.sca(ax)
    ax.set_aspect('equal')
    ax.set_xlim([limits[0], limits[1]])
    ax.set_ylim([limits[0], limits[1]])
    ax.plot(traj[0], traj[1], color="b")
    plot_frame_2d(frames.values())
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mode == 1:
                Z[i, j] = model.compute_potential_field(np.array([X[i, j], Y[i, j]]))
            else:
                Z[i, j] = model.compute_dissipation_field(np.array([X[i, j], Y[i, j]]))
    c = ax.pcolormesh(X, Y, Z, cmap=colormap, shading='auto', vmin=0., vmax=Z.max(), alpha=0.5)
    fig.colorbar(c, ax=ax)
    mcircle = plt.Circle(origin, moving_goal_radius, color='k', fill=False, linestyle='--', alpha=0.7)
    ax.add_patch(mcircle)
    ocircle = plt.Circle(cpose, R, color='k', fill=False)
    ax.add_patch(ocircle)

for ax in fig.axes:
    try:
        ax.label_outer()
    except:  # noqa
        pass
plt.show()
