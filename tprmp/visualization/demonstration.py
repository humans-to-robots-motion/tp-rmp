import numpy as np
import matplotlib.pyplot as plt
from tprmp.demonstrations.base import Demonstration
from tprmp.demonstrations.frame import Frame
from tprmp.demonstrations.quaternion import q_to_rotation_matrix
import logging

logger = logging.getLogger(__name__)


# TODO: color demo points following gamma
def plot_demo(demo, **kwargs):
    only_global = kwargs.get('only_global', True)
    new_fig = kwargs.get('new_fig', False)
    show = kwargs.get('show', False)
    if new_fig:
        plt.figure()
    if isinstance(demo, Demonstration):
        demo = [demo]
    _plot_traj_global(demo, **kwargs)
    if not only_global and len(demo[0].frame_names) > 0:
        plt.figure()  # new figure for other frames visual
        _plot_traj_frames(demo, **kwargs)
    if show:
        plt.show()


def _plot_traj_global(demos, **kwargs):
    legend = kwargs.get('legend', True)
    plot_frames = kwargs.get('plot_frames', True)
    title = kwargs.get('title', 'Global frame')
    new_ax = kwargs.get('new_ax', False)
    three_d = kwargs.get('three_d', True)
    if new_ax:
        if three_d:
            ax = plt.subplot(111, projection="3d")
        else:
            ax = plt.subplot(111)
    else:
        ax = plt.gca()
    plt.title(title)
    demo_tags = list(set([demo.tag for demo in demos]))
    tag_map = {v: i for i, v in enumerate(demo_tags)}
    cycle = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    # equal xyz scale
    sample_traj = demos[0].traj
    _equalize_axes(ax, sample_traj, three_d=three_d)
    for d in range(len(demos)):
        _plot_traj(demos[d].traj, label=demos[d].tag, color=cycle[tag_map[demos[d].tag]], **kwargs)
        if plot_frames:
            if three_d:
                plot_frame(demos[d].get_task_parameters().values())
            else:
                plot_frame_2d(demos[d].get_task_parameters().values())
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        temp = {k: v for k, v in zip(labels, handles)}
        plt.legend(temp.values(), temp.keys(), loc='best')


def _plot_traj_frames(demos, axs=None, **kwargs):
    legend = kwargs.get('legend', True)
    plot_frames = kwargs.get('plot_frames', True)
    three_d = kwargs.get('three_d', True)
    frames = demos[0].frame_names
    if axs is None:
        plt.clf()
        axs = {}
        for i, frame in enumerate(frames):
            if three_d:
                axs[frame] = plt.subplot(1, demos[0].num_frames, i + 1, projection="3d")
            else:
                axs[frame] = plt.subplot(1, demos[0].num_frames, i + 1)
            plt.title(f'Frame {frame}')
    for f_key in frames:
        plt.sca(axs[f_key])
        sample_traj = demos[0].traj_in_frames[f_key]['traj']
        _equalize_axes(axs[f_key], sample_traj, three_d=three_d)
        demo_tags = list(set([demo.tag for demo in demos]))
        tag_map = {v: i for i, v in enumerate(demo_tags)}
        cycle = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
        for d in range(len(demos)):
            _plot_traj(demos[d].traj_in_frames[f_key]['traj'], label=demos[d].tag, color=cycle[tag_map[demos[d].tag]], **kwargs)
        if plot_frames:
            if three_d:
                plt.plot([0, 1 / 40], [0, 0], [0, 0], 'r')
                plt.plot([0, 0], [0, 1 / 40], [0, 0], 'b')
                plt.plot([0, 0], [0, 0], [0, 1 / 40], 'g')
            else:
                plt.plot([0, 1 / 2.], [0, 0], 'r')
                plt.plot([0, 0], [0, 1 / 2.], 'b')
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        temp = {k: v for k, v in zip(labels, handles)}
        plt.legend(temp.values(), temp.keys(), loc='best')


def _plot_traj(traj, **kwargs):
    plot_quat = kwargs.get('plot_quat', True)
    label = kwargs.get('label', '')
    color = kwargs.get('color', 'b')
    skip_quat = kwargs.get('skip_quat', 4)
    three_d = kwargs.get('three_d', True)
    if three_d:
        plt.plot(traj[0, :], traj[1, :], traj[2, :], color=color, label=label)
    else:
        plt.plot(traj[0, :], traj[1, :], color=color, label=label)
    if three_d and plot_quat:
        for t in range(0, traj.shape[1], skip_quat):
            plot_frame(Frame(q_to_rotation_matrix(traj[-4:, t]), traj[:3, t]), length_scale=0.02, alpha=0.8)


def _equalize_axes(ax, traj, three_d=True, margin=0.05):
    X, Y = traj[0, :], traj[1, :]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max()
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    if three_d:
        Z = traj[2, :]
        max_range = max(max_range, Z.max() - Z.min())
        mid_z = (Z.max() + Z.min()) * 0.5
    max_range /= 2.0
    max_range += margin
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    if three_d:
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_frame(frame, length_scale=0.05, alpha=1.):
    if isinstance(frame, Frame):
        frame = [frame]
    ax = plt.gca()
    for f in frame:
        A = f.A
        b = f.b
        x_goal = b[:3] + A[:3, 0] * length_scale
        y_goal = b[:3] + A[:3, 1] * length_scale
        z_goal = b[:3] + A[:3, 2] * length_scale
        ax.plot([b[0], x_goal[0]], [b[1], x_goal[1]], [b[2], x_goal[2]], 'r', alpha=alpha)
        ax.plot([b[0], y_goal[0]], [b[1], y_goal[1]], [b[2], y_goal[2]], 'g', alpha=alpha)
        ax.plot([b[0], z_goal[0]], [b[1], z_goal[1]], [b[2], z_goal[2]], 'b', alpha=alpha)


def plot_frame_2d(frame, length_scale=0.05, alpha=1.):
    if isinstance(frame, Frame):
        frame = [frame]
    ax = plt.gca()
    for f in frame:
        A = f.A
        b = f.b
        x_goal = b[:2] + A[:2, 0] * length_scale
        y_goal = b[:2] + A[:2, 1] * length_scale
        ax.plot([b[0], x_goal[0]], [b[1], x_goal[1]], 'r', alpha=alpha)
        ax.plot([b[0], y_goal[0]], [b[1], y_goal[1]], 'g', alpha=alpha)
