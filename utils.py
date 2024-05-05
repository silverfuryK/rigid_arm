import numpy as np
import mujoco

def quaternion_to_rotation_matrix(quaternion):
    
    """Convert a quaternion to rotation matrix."""

    quaternion = quaternion.flatten()
    
    mat = np.empty((9, ), dtype=np.float64)
    mujoco._functions.mju_quat2Mat(mat, quaternion)
    return mat.reshape((3, 3))

def compute_relative_transformation(R1, t1, R2, t2):
    """Compute the transformation matrix of pose P2 relative to P1.
    
    Args:
    q1, t1: rotation matrix and translation (x, y, z) for pose P1.
    q2, t2: rotation matrix and translation (x, y, z) for pose P2.
    
    Returns:
    A 4x4 numpy array representing the transformation matrix of P2 relative to P1.
    """
    # Convert quaternions to rotation matrices

    
    # Create transformation matrices
    R1 = np.reshape(R1, (3, 3))
    R2 = np.reshape(R2, (3, 3))

    if np.linalg.det(R1) == 0:
        print("R1 is singular")
        R1 = np.eye(3)
    if np.linalg.det(R2) == 0:
        print("R2 is singular")
        R2 = np.eye(3)

    t_rel = t2 - t1

    T1 = np.eye(4)
    T2 = np.eye(4)
    T1[:3, :3], T1[:3, 3], T1[3, 3] = R1, t1, 1
    T2[:3, :3], T2[:3, 3], T2[3, 3] = R2, t2, 1
    
    # Compute the transformation of P2 relative to P1
    # print(T2)
    T1_inv = np.linalg.inv(T1)
    T2_1 = np.dot(T1_inv, T2)

    # print(T2_r_n)
    
    return T2_1

def rotation_matrix_to_quaternion(mat):
    """Convert a rotation matrix to a quaternion."""

    mat = mat.flatten()
    
    q = np.empty((4, ), dtype=np.float64)
    mujoco._functions.mju_mat2Quat(q, mat)
    return q

def recover_pose(T):
    """Recover quaternion and translation from a transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    q = rotation_matrix_to_quaternion(R)
    return q, t
