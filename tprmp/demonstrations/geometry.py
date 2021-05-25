import numpy as np
from scipy.linalg import block_diag


# ---------------------------------------
# free space transform (6-DOFs)
# ---------------------------------------
def construct_transform(p, q, q_rot_mat):
    """
    Construct orientation matrix A and translation vector b.

    Parameters
    ----------
    :param p (np.array): of shape (3,) for pose [x,y,z].
    :param q (np.array): of shape (4,) for quaternion [w,x,y,z].
    :param q_rot_mat (np.array): of shape (3,3) for the rotation matrix associated with q.

    Returns
    ----------
    :return A (np.array): of shape (6, 6), rotation in tangent space
    :return b (np.array): of shape (7,), translation in manifold space
    """
    A = block_diag(q_rot_mat, np.eye(3))
    b = np.append(p, q)
    return A, b
