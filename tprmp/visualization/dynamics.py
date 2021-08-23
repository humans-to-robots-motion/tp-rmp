import numpy as np
import matplotlib.pyplot as plt
from tprmp.visualization.demonstration import plot_frame_2d
from tprmp.visualization.models import _plot_gmm_global, _plot_gmm_frames


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
        ddx = tprmp.retrieve(x, dx, frames)
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


def plot_potential_field(tprmp, frames, **kwargs):
    only_global = kwargs.get('only_global', True)
    plot_gaussian = kwargs.get('plot_gaussian', True)
    three_d = kwargs.get('three_d', False)
    margin = kwargs.get('margin', 0.5)
    res = kwargs.get('res', 0.1)
    new_fig = kwargs.get('new_fig', False)
    show = kwargs.get('show', False)
    origin = tprmp.model.manifold.get_origin()
    frame_origins = np.array([v.transform(origin) for v in frames.values()])
    ranges = np.array([frame_origins[:, 0].max() - frame_origins[:, 0].min(), frame_origins[:, 1].max() - frame_origins[:, 1].min()]).max() * 0.5 + margin
    mid_x = (frame_origins[:, 0].max() + frame_origins[:, 0].min()) * 0.5
    mid_y = (frame_origins[:, 1].max() + frame_origins[:, 1].min()) * 0.5
    if new_fig:
        plt.figure()
    _plot_potential_field_global(tprmp, frames, [mid_x, mid_y], ranges, plot_gaussian=plot_gaussian, three_d=three_d, res=res)
    if not only_global:
        if three_d:
            plt.figure()
        _plot_potential_field_frames(tprmp, frames, ranges, plot_gaussian=plot_gaussian, three_d=three_d, res=res)
    if show:
        plt.show()


def _plot_potential_field_global(tprmp, frames, mid, ranges, plot_gaussian=True, three_d=False, res=0.1, alpha=0.7):
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
            Z[i, j] = tprmp.compute_potential_field(np.array([X[i, j], Y[i, j]]))
    if three_d:
        c = ax.plot_surface(X, Y, Z, cmap='RdBu', vmin=0., vmax=Z.max(), alpha=alpha)
    else:
        c = ax.pcolormesh(X, Y, Z, cmap='RdBu', shading='auto', vmin=0., vmax=Z.max(), alpha=alpha)
        ax.axes.set_aspect('equal')
    ax.set_title('Global potential Phi')
    plt.gcf().colorbar(c, ax=ax)
    plot_frame_2d(frames.values())
    if plot_gaussian:
        _plot_gmm_global(tprmp.model, frames, three_d=False, new_ax=False)


def _plot_potential_field_frames(tprmp, frames, ranges, axs=None, plot_gaussian=True, three_d=False, res=0.1, alpha=0.7):
    if axs is None:
        axs = {}
        if three_d:
            plt.clf()
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
                Z[f_key][i, j] = tprmp.compute_potential_field_frame(np.array([X[i, j], Y[i, j]]), f_key)
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
        _plot_gmm_frames(tprmp.model, frames, axs=axs, three_d=False)
