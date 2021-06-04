import numpy as np

'''
"Programming by Demonstration on Riemannian Manifolds", M.J.A. Zeestraten, 2018.
Format: q = [w, x, y, z]. # NOTE change to [x, y, z, w] for pyBullet
'''


def q_exp_map(v, base=None):
    v_2d = v.reshape((3, 1)) if len(v.shape) == 1 else v
    if base is None:
        norm_v = np.sqrt(np.sum(v_2d**2, axis=0))
        q = np.append(np.ones((1, v_2d.shape[1])), np.zeros((3, v_2d.shape[1])), axis=0)
        non_0 = np.where(norm_v > 0)[0]
        q[:, non_0] = np.append(
            np.cos(norm_v[non_0]).reshape((1, non_0.shape[0])),
            np.tile(np.sin(norm_v[non_0]) / norm_v[non_0], (3, 1)) * v_2d[:, non_0], axis=0)
        return q.reshape(4) if len(v.shape) == 1 else q
    else:
        return q_mul(base, q_exp_map(v))


def q_log_map(q, base=None):
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    if base is None:
        norm_q = np.sqrt(np.sum(q_2d[1:, :]**2, axis=0))
        non_0 = np.where((norm_q > 0) * (np.abs(q_2d[0, :]) <= 1))[0]
        q_non_singular = q_2d[:, non_0]
        acos = np.arccos(q_non_singular[0, :])
        acos[np.where(q_non_singular[0, :] < 0)] += -np.pi  # q and -q maps are the same
        v = np.zeros((3, q_2d.shape[1]))
        v[:, non_0] = q_non_singular[1:, :] * np.tile(acos / norm_q[non_0], (3, 1))
        if len(q.shape) == 1:
            return v.reshape(3)
        return v
    else:
        return q_log_map(q_mul(q_inverse(base), q))


def q_parallel_transport(p_g, g, h):
    R_e_g = q_to_quaternion_matrix(g)
    R_h_e = q_to_quaternion_matrix(h).T
    B = np.append(np.zeros((3, 1)), np.eye(3), 1).T
    log_g_h = q_log_map(h, base=g)
    m = np.linalg.norm(log_g_h)
    if m < 1e-10:  # divide by zero
        return p_g
    u = R_e_g.dot(np.append(0, log_g_h / m)).reshape((4, 1))
    R_g_h = np.eye(4) - np.sin(m) * g.reshape((4, 1)).dot(u.T) + (np.cos(m) - 1) * u.dot(u.T)
    A_g_h = B.T.dot(R_h_e).dot(R_g_h).dot(R_e_g).dot(B)
    return A_g_h.dot(p_g)


def q_mul(q1, q2):
    return q_to_quaternion_matrix(q1).dot(q2)


def q_inverse(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / q_norm_squared(q)


def q_div(q1, q2):
    return q_mul(q1, q_inverse(q2))


def q_norm_squared(q):
    return np.sum(q**2)


def q_norm(q):
    return np.sqrt(q_norm_squared(q))


def q_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]])


def q_to_quaternion_matrix(q):
    w, x, y, z = q
    return np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])


def q_from_rot_mat(rot_mat):
    qs = min(np.sqrt(np.trace(rot_mat) + 1) / 2.0, 1.0)
    kx = rot_mat[2, 1] - rot_mat[1, 2]  # Oz - Ay
    ky = rot_mat[0, 2] - rot_mat[2, 0]  # Ax - Nz
    kz = rot_mat[1, 0] - rot_mat[0, 1]  # Ny - Ox
    if (rot_mat[0, 0] >= rot_mat[1, 1]) and (rot_mat[0, 0] >= rot_mat[2, 2]):
        kx1 = rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2] + 1  # Nx - Oy - Az + 1
        ky1 = rot_mat[1, 0] + rot_mat[0, 1]  # Ny + Ox
        kz1 = rot_mat[2, 0] + rot_mat[0, 2]  # Nz + Ax
        plus = (kx >= 0)
    elif rot_mat[1, 1] >= rot_mat[2, 2]:
        kx1 = rot_mat[1, 0] + rot_mat[0, 1]  # Ny + Ox
        ky1 = rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2] + 1  # Oy - Nx - Az + 1
        kz1 = rot_mat[2, 1] + rot_mat[1, 2]  # Oz + Ay
        plus = (ky >= 0)
    else:
        kx1 = rot_mat[2, 0] + rot_mat[0, 2]  # Nz + Ax
        ky1 = rot_mat[2, 1] + rot_mat[1, 2]  # Oz + Ay
        kz1 = rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1] + 1  # Az - Nx - Oy + 1
        plus = (kz >= 0)
    if plus:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1
    nm = np.linalg.norm(np.array([kx, ky, kz]))
    if nm == 0:
        q = np.array([1, 0, 0, 0])
    else:
        s = np.sqrt(1 - qs**2) / nm
        qv = s * np.array([kx, ky, kz])
        q = np.append(qs, qv)
    return q


def q_to_euler(q):
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    w, x, y, z = q_2d[0, :], q_2d[1, :], q_2d[2, :], q_2d[3, :]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    euler = np.stack([roll, pitch, yaw], axis=0)
    return euler


def q_from_euler(euler):
    euler_2d = euler.reshape((4, 1)) if len(euler.shape) == 1 else euler
    roll, pitch, yaw = euler_2d[0, :], euler_2d[1, :], euler_2d[2, :]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.stack([w, x, y, z], axis=0)
    return q


def q_convert_xyzw(q):
    q_2d = q.reshape((4, 1)) if len(q.shape) == 1 else q
    w, x, y, z = q_2d[0, :], q_2d[1, :], q_2d[2, :], q_2d[3, :]
    return np.stack([x, y, z, w], axis=0)
