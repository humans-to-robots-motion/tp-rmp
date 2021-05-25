import numpy as np


def q_to_rotation_matrix(q):
    """
    Computes rotation matrix out of the quaternion.

    Parameters
    ----------
    :param q: np.array of shape (4,), the quaternion

    Returns
    ----------
    :return rot_mat: np.array of shape (3, 3)
    """
    w, x, y, z = q
    return np.array([[1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]])
