import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import logging

from tprmp.visualization.demonstration import plot_frame, plot_frame_2d

logger = logging.getLogger(__name__)


def plot_gmm(model, frames, only_global=True, legend=True, new_fig=False, show=False, **kwargs):
    if new_fig:
        plt.figure()
    _plot_gmm_global(model, frames, **kwargs)
    if not only_global and len(frames) > 0:
        plt.figure()
        _plot_gmm_frames(model, frames, **kwargs)
    if legend:
        plt.legend()
    if show:
        plt.show()


def _plot_gmm_global(model, frames, **kwargs):
    var_scale = kwargs.get('var_scale', 1.)  # to plot descale variances
    plot_quat = kwargs.get('plot_quat', False)
    plot_frames = kwargs.get('plot_frames', False)
    tag = kwargs.get('tag', None)
    three_d = kwargs.get('three_d', True)
    new_ax = kwargs.get('new_ax', True)
    if new_ax:
        if three_d:
            plt.subplot(111, projection="3d")
        else:
            plt.subplot(111)
    global_mvns = model.generate_global_gmm(frames, tag=tag)
    cycle = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    for k in range(len(global_mvns)):  # TODO: change to multiple cluster with tags
        if isinstance(var_scale, list):
            scale = var_scale[k]
        else:
            scale = var_scale
        _plot_gaussian(global_mvns[k], color=cycle[k % len(cycle)], three_d=three_d, var_scale=scale)
        if three_d and plot_quat:
            plot_frame(global_mvns[k].mean[-4:], global_mvns[k].mean[:3], length_scale=0.02, alpha=0.8)
    if plot_frames:
        if three_d:
            plot_frame(frames.values())
        else:
            plot_frame_2d(frames.values())


def _plot_gmm_frames(model, frames, axs=None, **kwargs):
    var_scale = kwargs.get('var_scale', 1.)  # to plot descale variances
    plot_quat = kwargs.get('plot_quat', False)
    plot_frames = kwargs.get('plot_frames', False)
    tag = kwargs.get('tag', None)
    three_d = kwargs.get('three_d', True)
    if axs is None:
        plt.clf()
        axs = {}
        for i, frame in enumerate(frames):
            if three_d:
                axs[frame] = plt.subplot(1, model.num_frames, i + 1, projection="3d")
            else:
                axs[frame] = plt.subplot(1, model.num_frames, i + 1)
            plt.title(f'Frame {frame}')
    cycle = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    for frame in frames:
        plt.sca(axs[frame])
        comps = range(model.num_comp) if ((model.tag_to_comp_map is None) or
                                          (tag not in model.tag_to_comp_map)) else model.tag_to_comp_map[tag]
        for k in comps:  # TODO: change to multiple cluster with tags
            if isinstance(var_scale, list):
                scale = var_scale[k]
            else:
                scale = var_scale
            _plot_gaussian(model.mvns[k][frame], color=cycle[k % len(cycle)], three_d=three_d, var_scale=scale)
            if three_d and plot_quat:
                plot_frame(model.mvns[k][frame].mean[-4:], model.mvns[k][frame].mean[:3], length_scale=0.02, alpha=0.8)
        if plot_frames:
            if three_d:
                plt.plot([0, 1 / 40], [0, 0], [0, 0], 'r')
                plt.plot([0, 0], [0, 1 / 40], [0, 0], 'b')
                plt.plot([0, 0], [0, 0], [0, 1 / 40], 'g')
            else:
                plt.plot([0, 1 / 10.], [0, 0], 'r')
                plt.plot([0, 0], [0, 1 / 10.], 'b')


def _plot_gaussian(mvn, color='b', three_d=True, var_scale=1.):
    if three_d:
        mu = mvn.mean[:3]
        cov = mvn.cov[:3, :3] / (var_scale**2)
        center = mu[0:3]
        _, s, rotation = np.linalg.svd(cov)
        radii = np.sqrt(s)
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
        plt.gca().plot_wireframe(x, y, z, rstride=4, cstride=4, alpha=0.3, color=color)
        plt.plot([mu[0]], [mu[1]], [mu[2]], marker='o', color=color, markersize=1)
    else:
        mu = mvn.mean[:2]
        cov = mvn.cov[:2, :2] / (var_scale**2)
        w, v = np.linalg.eig(cov)
        x, y = v[:, 0]
        theta = float(np.degrees(np.arctan2(y, x)))
        width, height = 2 * np.sqrt(w)
        ellipse = Ellipse(mu[0:2], width, height, angle=theta, color=color, fill=True)
        ax = plt.gca()
        ax.add_patch(ellipse)
        if ax.name == '3d':
            art3d.pathpatch_2d_to_3d(ellipse, z=0, zdir='z')
        plt.plot([mu[0]], [mu[1]], marker='o', color=color, markersize=1)


def plot_hsmm(model, end_states=True, legend=True, duration=True, new_fig=False, show=False):  # TODO: check plotting locations
    if new_fig:
        plt.figure()
    comps = range(model.num_comp)
    clusters = [np.nonzero(model.pi > 1e-5)[0].tolist()]
    visited = list(clusters[0])
    for k in comps:
        clusters.append([])
        for i in clusters[-2]:
            next_states = np.nonzero(model.trans_prob[i, :] > 1e-5)[0].tolist()
            for state in next_states:
                if state not in visited:
                    clusters[k + 1].append(state)
                    visited.append(state)
        if len(clusters[-1]) == 0:
            clusters.remove(clusters[-1])
            comps = visited
            logger.warn("Some states of the HSMM are unreachable!")
            break
        if len(visited) == model.num_comp:
            break
    clusters.reverse()
    cycle = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    ax = plt.gca()
    x0, y0, w, h = ax.get_position().bounds
    ax.axis('off')
    ax.set_title(f'The HSMM model of {model.name}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    y_spread = np.linspace(0, 1, len(clusters) + 2)
    loc = [None] * model.num_comp
    for ky in range(len(clusters)):
        x_spread = np.linspace(0, 1, len(clusters[ky]) + 2)
        for kx in range(len(clusters[ky])):
            loc[clusters[ky][kx]] = np.array([x_spread[kx + 1], y_spread[ky + 1]])
    for i in comps:
        if i in clusters[-1]:
            x = loc[i] + 0.12 * np.array([-1, 1])
            dx = 0.1 * np.array([1, -1])
            ax.arrow(x[0], x[1], dx[0], dx[1], color='k', head_width=0.01, length_includes_head=True)
            ax.text(x[0] + 0.5 * dx[0] + 0.01, x[1] + 0.5 * dx[1], "%.2g" % model.pi[i])
        ax.plot(loc[i][0], loc[i][1], 'o', color=cycle[i % len(cycle)], markersize=10, label=model.component_names[i])
        for j in comps:
            if model.trans_prob[i, j] > 1e-5 and j != i:  # there is a transition
                x = 0.9 * loc[i] + 0.1 * loc[j]
                dx = 0.1 * loc[i] + 0.9 * loc[j] - x
                if dx[1] > 0:
                    x[0] += 0.01
                elif dx[1] == 0:
                    if dx[0] > 0:
                        x[1] += 0.005
                    else:
                        x[1] -= 0.005
                ax.arrow(x[0], x[1], dx[0], dx[1], color='k', head_width=0.01, length_includes_head=True)
                ax.text(x[0] + 0.5 * dx[0] + 0.01, x[1] + 0.5 * dx[1], "%.2g" % model.trans_prob[i, j])
        # plot duration
        if duration:
            if i in clusters[0]:
                a = ax.get_figure().add_axes([x0 + (loc[i][0] - 0.09) * w, y0 + (loc[i][1] - 0.12) * h, .1 * w, .1 * h], xticks=[], yticks=[])
            elif i in clusters[-1]:
                a = ax.get_figure().add_axes([x0 + (loc[i][0] - 0.01) * w, y0 + (loc[i][1] + 0.02) * h, .1 * w, .1 * h], xticks=[], yticks=[])
            elif i in [layer[0] for layer in clusters]:
                a = ax.get_figure().add_axes([x0 + (loc[i][0] - 0.12) * w, y0 + (loc[i][1] - 0.05) * h, .1 * w, .1 * h], xticks=[], yticks=[])
            else:
                a = ax.get_figure().add_axes([x0 + (loc[i][0] + 0.02) * w, y0 + (loc[i][1] - 0.05) * h, .1 * w, .1 * h], xticks=[], yticks=[])
            a.plot(model.duration_prob[i, :], color=cycle[i % len(cycle)])
        # plot end states
        if end_states:
            if model.end_states[i] > 1e-3:
                x = loc[i] + 0.02 * np.array([1, -1])
                dx = 0.1 * np.array([1, -1])
                ax.text(x[0] + 0.5 * dx[0] + 0.01, x[1] + 0.5 * dx[1], "End: %.2g" % model.end_states[i])
    if legend:
        ax.legend()
    if show:
        plt.show()
