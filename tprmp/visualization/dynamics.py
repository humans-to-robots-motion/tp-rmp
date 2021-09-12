from tprmp.demonstrations.manifold import Manifold
import numpy as np
import matplotlib.pyplot as plt
from tprmp.visualization.demonstration import plot_frame_2d, plot_frame
from tprmp.models.rmp import compute_obsrv_prob, compute_pulls, compute_potentials
from tprmp.visualization.models import _plot_gmm_global, _plot_gmm_frames
import matplotlib
from matplotlib import cm
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def visualize_rmp(tprmp, frames, x0, dx0, T, dt, sample=None, x_limits=[0., 5.], vel_limits=[-10., 10.]):  # TODO: extend to 3D later
    plt.ion()
    x, dx = x0, dx0
    traj, traj_vel = [x], [dx]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    line1, = ax1.plot(x0[0], x0[1], marker="o", color="b", markersize=1, linestyle='None', alpha=1.)
    line2, = ax1.plot(x0[0], x0[1], marker="o", color="b", markersize=1, linestyle='None', alpha=0.3)
    vel1, = ax2.plot(0, dx0[0], color='r')
    vel2, = ax2.plot(0, dx0[1], color='g')
    if sample is not None:
        length = sample.traj.shape[1]
        ax1.plot(sample.traj[0], sample.traj[1], color="b", linestyle='--', alpha=0.6)
        ax2.plot(range(length), sample.d_traj[0], color="r", linestyle='--', alpha=0.6)
        ax2.plot(range(length), sample.d_traj[1], color="g", linestyle='--', alpha=0.6)
    ax1.set_xlim([x_limits[0], x_limits[1]])
    ax1.set_ylim([x_limits[0], x_limits[1]])
    ax1.set_aspect('equal', 'box')
    ax2.legend([vel1, vel2], ['vx', 'vy'])
    ax2.set_xlim([0, T])
    ax2.set_ylim([vel_limits[0], vel_limits[1]])
    tprmp.generate_global_gmm(frames)
    for t in range(1, T):
        ddx = tprmp.retrieve(x, dx)
        dx = ddx * dt + dx
        x = dx * dt + x
        traj.append(x)
        traj_vel.append(dx)
        # redraw
        line1.set_xdata(x[0])
        line1.set_ydata(x[1])
        line2.set_xdata(np.array(traj)[:, 0])
        line2.set_ydata(np.array(traj)[:, 1])
        vel1.set_xdata(range(t + 1))
        vel1.set_ydata(np.array(traj_vel)[:, 0])
        vel2.set_xdata(range(t + 1))
        vel2.set_ydata(np.array(traj_vel)[:, 1])
        fig.canvas.draw()
        fig.canvas.flush_events()


def plot_heatmap_3d(tprmp, frames, **kwargs):
    '''3D heatmap using scatter'''
    only_global = kwargs.get('only_global', True)
    plot_gaussian = kwargs.get('plot_gaussian', True)
    margin = kwargs.get('margin', 0.5)
    res = kwargs.get('res', 0.1)
    new_fig = kwargs.get('new_fig', False)
    show = kwargs.get('show', False)
    mid, ranges = _get_data_ranges(frames, tprmp.model.manifold.get_origin(), margin=margin, d=3)
    if new_fig:
        plt.figure()
    _plot_heatmap_3d_global(tprmp, frames, mid, ranges, plot_gaussian=plot_gaussian, res=res)
    if not only_global:
        pass  # TODO: implement plotting in frame if needed
    if show:
        plt.show()


def _plot_heatmap_3d_global(tprmp, frames, mid, ranges, plot_gaussian=True, res=0.1, alpha=0.5):
    ax = plt.subplot(111, projection="3d")
    x = np.arange(mid[0] - ranges, mid[0] + ranges, res)
    y = np.arange(mid[1] - ranges, mid[1] + ranges, res)
    z = np.arange(mid[2] - ranges, mid[2] + ranges, res)
    manifold = Manifold.get_euclidean_manifold(3)
    X, Y, Z = np.meshgrid(x, y, z)
    V = np.zeros_like(Z)
    tprmp.generate_global_gmm(frames)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                V[i, j, k] = tprmp.compute_potential_field(np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]]), manifold=manifold)
    color_map = cm.ScalarMappable(cmap='RdBu')
    color_map.set_array(np.ravel(V))
    ax.scatter(np.ravel(X), np.ravel(Y), np.ravel(Z), s=10, cmap='RdBu', alpha=alpha)
    ax.set_title('Global potential field')
    plt.colorbar(color_map)
    plot_frame(frames.values())
    if plot_gaussian:
        _plot_gmm_global(tprmp.model, frames, three_d=True, new_ax=False)


def plot_dissipation_field(tprmp, frames, **kwargs):
    '''Only works with 2D data'''
    only_global = kwargs.pop('only_global', True)
    margin = kwargs.pop('margin', 0.5)
    limits = kwargs.pop('limits', None)
    new_fig = kwargs.pop('new_fig', False)
    show = kwargs.pop('show', False)
    if limits is None:
        mid, ranges = _get_data_ranges(frames, tprmp.model.manifold.get_origin(), margin=margin)
    else:
        m = (limits[1] + limits[0]) / 2.
        mid = [m, m]
        ranges = (limits[1] - limits[0]) / 2.
    if new_fig:
        plt.figure()
    _plot_dissipation_field_global(tprmp, frames, mid, ranges, **kwargs)
    if not only_global:
        pass  # TODO: implement plotting in frame if needed
    if show:
        plt.show()


def _plot_dissipation_field_global(tprmp, frames, mid, ranges, plot_gaussian=True, var_scale=1., res=0.1, alpha=0.7):
    ax = plt.subplot(111)
    x = np.arange(mid[0] - ranges, mid[0] + ranges, res)
    y = np.arange(mid[1] - ranges, mid[1] + ranges, res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    tprmp.generate_global_gmm(frames)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = tprmp.compute_dissipation_field(np.array([X[i, j], Y[i, j]]))
    c = ax.pcolormesh(X, Y, Z, cmap='YlOrBr', shading='auto', vmin=0., vmax=Z.max(), alpha=alpha)
    ax.axes.set_aspect('equal')
    ax.set_title('Global dissipative field')
    plt.gcf().colorbar(c, ax=ax)
    plot_frame_2d(frames.values())
    if plot_gaussian:
        _plot_gmm_global(tprmp.model, frames, var_scale=var_scale, three_d=False, new_ax=False)


def plot_potential_field(tprmp, frames, **kwargs):
    '''Only works with 2D data'''
    only_global = kwargs.pop('only_global', True)
    margin = kwargs.pop('margin', 0.5)
    limits = kwargs.pop('limits', None)
    new_fig = kwargs.pop('new_fig', False)
    show = kwargs.pop('show', False)
    if limits is None:
        mid, ranges = _get_data_ranges(frames, tprmp.model.manifold.get_origin(), margin=margin)
    else:
        m = (limits[1] + limits[0]) / 2.
        mid = [m, m]
        ranges = (limits[1] - limits[0]) / 2.
    if new_fig:
        plt.figure()
    _plot_potential_field_global(tprmp, frames, mid, ranges, **kwargs)
    if not only_global:
        plt.figure()
        _plot_potential_field_frames(tprmp, frames, ranges, **kwargs)
    if show:
        plt.show()


def _plot_potential_field_global(tprmp, frames, mid, ranges, plot_gaussian=True, var_scale=1., three_d=False, res=0.1, max_z=1000, alpha=0.7):
    if three_d:
        ax = plt.subplot(111, projection='3d')
    else:
        ax = plt.subplot(111)
    x = np.arange(mid[0] - ranges, mid[0] + ranges, res)
    y = np.arange(mid[1] - ranges, mid[1] + ranges, res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    tprmp.generate_global_gmm(frames)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = min(tprmp.compute_potential_field(np.array([X[i, j], Y[i, j]])), max_z)  # constraining plot
    if three_d:
        c = ax.plot_surface(X, Y, Z, cmap='RdBu', vmin=0., vmax=Z.max(), alpha=alpha)
    else:
        c = ax.pcolormesh(X, Y, Z, cmap='RdBu', shading='auto', vmin=0., vmax=Z.max(), alpha=alpha)
        ax.axes.set_aspect('equal')
    ax.set_title('Global potential field')
    plt.gcf().colorbar(c, ax=ax)
    plot_frame_2d(frames.values())
    if plot_gaussian:
        _plot_gmm_global(tprmp.model, frames, var_scale=var_scale, three_d=False, new_ax=False)


def _plot_potential_field_frames(tprmp, frames, ranges, axs=None, plot_gaussian=True, var_scale=1., three_d=False, res=0.1, max_z=1000, alpha=0.7):
    if axs is None:
        axs = {}
        if three_d:
            for i, frame in enumerate(frames):
                axs[frame] = plt.subplot(1, len(frames), i + 1, projection="3d")
        else:
            _, axes = plt.subplots(1, len(frames), figsize=(14, 6))
            for i, frame in enumerate(frames):
                axs[frame] = axes[i]
    x = y = np.arange(-ranges * 2, ranges * 2, res)
    X, Y = np.meshgrid(x, y)
    Z = {}
    z_max = 0.
    for f_key in frames:
        Z[f_key] = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[f_key][i, j] = min(tprmp.compute_potential_field_frame(np.array([X[i, j], Y[i, j]]), f_key), max_z)
        z_max = max(z_max, Z[f_key].max())
    for f_key in frames:
        if three_d:
            c = axs[f_key].plot_surface(X, Y, Z[f_key], cmap='RdBu', vmin=0., vmax=z_max, alpha=alpha)
        else:
            c = axs[f_key].pcolormesh(X, Y, Z[f_key], cmap='RdBu', shading='auto', vmin=0., vmax=z_max, alpha=alpha)
            axs[f_key].axes.set_aspect('equal')
        axs[f_key].set_title(f'Frame {f_key}')
        plt.gcf().colorbar(c, ax=axs[f_key])
        axs[f_key].plot([0, 1 / 2], [0, 0], 'r')
        axs[f_key].plot([0, 0], [0, 1 / 2], 'b')
    if plot_gaussian:
        _plot_gmm_frames(tprmp.model, frames, axs=axs, var_scale=var_scale, three_d=False)


def _get_data_ranges(frames, origin, margin=0.1, d=2):
    frame_origins = np.array([v.transform(origin) for v in frames.values()])
    minmax = np.zeros((d, 2))
    for i in range(d):
        minmax[i] = [frame_origins[:d, i].min(), frame_origins[:d, i].max()]
    ranges = (minmax[:, 1] - minmax[:, 0]).max() * 0.5 + margin
    mid = (minmax[:, 1] + minmax[:, 0]) * 0.5
    return mid, ranges


def plot_potential_grad(tprmp, frames, sample=None, **kwargs):
    '''Only works with 2D data'''
    warped = kwargs.get('warped', False)
    plot_frames = kwargs.get('plot_frames', True)
    margin = kwargs.get('margin', 0.1)
    res = kwargs.get('res', 0.05)
    new_fig = kwargs.get('new_fig', False)
    colorbar = kwargs.get('colorbar', False)
    show = kwargs.get('show', False)
    mid, ranges = _get_data_ranges(frames, tprmp.model.manifold.get_origin(), margin=margin)
    x = np.arange(mid[0] - ranges, mid[0] + ranges, res)
    y = np.arange(mid[1] - ranges, mid[1] + ranges, res)
    X, Y = np.meshgrid(x, y)
    px, py = np.zeros_like(X), np.zeros_like(X)
    color = np.zeros_like(X)
    tprmp.generate_global_gmm(frames)
    if new_fig:
        plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = np.array([X[i, j], Y[i, j]])
            px[i, j], py[i, j] = tprmp.compute_potential_grad(p, warped=warped)
            color[i, j] = np.linalg.norm(np.array([px[i, j], py[i, j]]))
    c = ax.quiver(X, Y, px, py, color, headwidth=2, headlength=3)
    if sample is not None:
        ax.plot(sample.traj[0], sample.traj[1], color="b", linestyle='--', alpha=0.6)
    ax.set_aspect('equal')
    if colorbar:
        plt.gcf().colorbar(c, ax=ax)
    if plot_frames:
        plot_frame_2d(frames.values())
    if show:
        plt.show()


def plot_weight_map(tpgmm, frames, sample=None, **kwargs):
    '''Only works with 2D data'''
    plot_frames = kwargs.get('plot_frames', True)
    margin = kwargs.get('margin', 0.1)
    res = kwargs.get('res', 0.05)
    alpha = kwargs.get('alpha', 0.5)
    new_fig = kwargs.get('new_fig', False)
    show = kwargs.get('show', False)
    mid, ranges = _get_data_ranges(frames, tpgmm.manifold.get_origin(), margin=margin)
    x = np.arange(mid[0] - ranges, mid[0] + ranges, res)
    y = np.arange(mid[1] - ranges, mid[1] + ranges, res)
    X, Y = np.meshgrid(x, y)
    img = np.zeros((X.shape[0], X.shape[1], 3))
    mvns = tpgmm.generate_global_gmm(frames)
    if new_fig:
        plt.figure()
    cycle = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    rgb_values = hex_to_rgb(cycle[:tpgmm.num_comp])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = np.array([X[i, j], Y[i, j]])
            weights = compute_obsrv_prob(p, mvns)
            img[i, j] = weights.T @ rgb_values
    plt.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], interpolation='nearest', origin='lower', alpha=alpha)
    if sample is not None:
        plt.plot(sample.traj[0], sample.traj[1], color="b", linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal')
    if plot_frames:
        plot_frame_2d(frames.values())
    if show:
        plt.show()


def hex_to_rgb(colors):
    rgb_arr = []
    for c in colors:
        h = c.lstrip('#')
        rgb_arr.append(np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.)
    return np.array(rgb_arr)


def plot_potential_force_components(tprmp, frames, x, new_fig=False, show=False, scale=50.):
    '''Only works with 2D data, plotting only huber potential forces'''
    if new_fig:
        plt.figure()
    stiff_scale = tprmp._stiff_scale
    delta = tprmp._delta
    mvns = tprmp.model.generate_global_gmm(frames)
    num_comp = len(mvns)
    manifold = mvns[0].manifold
    pulls = compute_pulls(x, mvns)
    weights = compute_obsrv_prob(x, mvns)
    mean_pull = weights.T @ pulls
    phi = compute_potentials(tprmp.phi0, x, mvns, stiff_scale=stiff_scale, delta=delta)
    means = [mvns[k].mean for k in range(num_comp)]
    cycle = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    nominals = []
    attractors = []
    for k in range(num_comp):
        v = manifold.log_map(x, base=mvns[k].mean)
        norm = np.sqrt((stiff_scale**2) * v.T @ pulls[k])
        nominals.append(weights[k] * phi[k] * (pulls[k] - mean_pull))
        if norm <= delta:
            attractors.append(-weights[k] * (stiff_scale**2) * pulls[k])
        else:
            attractors.append(-weights[k] * (stiff_scale**2) * delta * pulls[k] / norm)
    attractors = np.array(attractors)
    nominals = np.array(nominals)
    means = np.array(means)
    nf = np.sum(nominals, axis=0)
    af = np.sum(attractors, axis=0)
    colors = hex_to_rgb(cycle[:num_comp]).tolist()
    plt.quiver(means[:, 0], means[:, 1], nominals[:, 0], nominals[:, 1], color=colors, headwidth=2, headlength=3, angles='xy', scale_units='xy', scale=scale)
    plt.quiver(means[:, 0], means[:, 1], attractors[:, 0], attractors[:, 1], color=colors, headwidth=2, headlength=3, hatch='+++', angles='xy', scale_units='xy', scale=scale-30)
    plt.quiver(x[0], x[1], nf[0], nf[1], color='0.5', headwidth=2, headlength=3, angles='xy', scale_units='xy', scale=scale)
    plt.quiver(x[0], x[1], af[0], af[1], color='0.5', headwidth=2, headlength=3, hatch='+++', angles='xy', scale_units='xy', scale=scale-30)
    plt.scatter(means[:, 0], means[:, 1], marker='o', color=cycle[:num_comp])
    plt.scatter(x[0], x[1], marker='o', color='k')
    if show:
        plt.show()
