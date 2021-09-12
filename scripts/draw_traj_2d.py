import sys
from matplotlib import pyplot as plt
from os.path import join, dirname, abspath

ROOT_DIR = join(dirname(abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

from tprmp.utils.recorder2d import Recorder2D  # noqa

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
ax.set_title('Left-click to start record, left-click again to stop')
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_aspect('equal')
p, = ax.plot([], [], linestyle="none", marker="o", color="r", markersize=1)
recorder = Recorder2D(p)

plt.show()
