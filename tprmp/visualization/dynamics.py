import numpy as np
import matplotlib.pyplot as plt
import torch
from tprmp.models.rmp import compute_policy


def visualize_rmp(phi0, d0, mvns, x0, dx0, T, dt, limit=10., R_net=None):  # TODO: extend to 3D later
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
    ax1.set_xlim([0, limit])
    ax1.set_ylim([0, limit])
    ax1.set_aspect('equal', 'box')
    ax2.legend([vel1, vel2], ['vx', 'vy'])
    ax2.set_xlim([0, T])
    ax2.set_ylim([0, limit])
    for t in range(1, T):
        if R_net is not None:
            x_t, dx_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0), torch.from_numpy(dx.astype(np.float32)).unsqueeze(0)
            M, M_inv, c, _ = R_net(x_t, dx_t)
            M_inv, c = M_inv.detach().cpu().squeeze().numpy(), c.detach().cpu().squeeze().numpy()
            ddx = M_inv @ (compute_policy(phi0, d0, x, dx, mvns) - c)
        else:
            ddx = compute_policy(phi0, d0, x, dx, mvns)
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
