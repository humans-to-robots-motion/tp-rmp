import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_gamma(gamma, new_fig=False, show=False):
    if new_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()  # context dependent
        ax = plt.gca()
    ax.set_axis_off()
    ax.set_title("Gamma")
    sps = ax.get_subplotspec()
    max_len = max([g.shape[0] for g in gamma])
    grid = gridspec.GridSpecFromSubplotSpec(len(gamma), 1, subplot_spec=sps, wspace=0.1, hspace=0.1)
    for m in range(len(gamma)):
        axk = fig.add_subplot(grid[m, 0])
        axk.set_xlim(0, max_len)
        for k in range(gamma[0].shape[1]):
            axk.plot(gamma[m][:, k])
    if show:
        plt.show()
