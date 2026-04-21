import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


def print_array(array, suppress_scientific=False, precision=8):
    """
    Prints a numpy array with a specified precision and optionally suppresses scientific notation.
    """
    if suppress_scientific:
        print(np.array2string(array, 
                              suppress_small=True, precision=precision))
    else:
        print(array)
        
def pose_to_TMatrix(pose: list[float]|npt.ArrayLike):
    """
    Converts a pose represented by a list or array-like object into a 
    4x4 transformation matrix.
    
    Args:
        pose : A list or array-like object containing six elements representing the 
        pose of the form [x, y, z, rx, ry, rz).
    """

    x, y, z, rx, ry, rz = pose
    rot = R.from_rotvec([rx, ry, rz])
    m = np.eye(4)
    m[:3, :3] = rot.as_matrix()
    m[:3, 3] = [x, y, z]
    return m

def TMatrix_to_pose(m: npt.NDArray):
    """
    Converts a 4x4 transformation matrix to a pose representation.

    Returns:
        A list containing the translation and rotation in the form 
        [tx, ty, tz, rx, ry, rz], where (tx, ty, tz) is the translation 
        vector and (rx, ry, rz) is the rotation vector in radians.
    """

    rot = R.from_matrix(m[:3, :3])
    t = m[:3, 3]
    rx, ry, rz = rot.as_rotvec()
    return [t[0], t[1], t[2], rx, ry, rz]

def inverse_TMatrix(T: npt.NDArray):
    """
    Compute the inverse of a 4x4 transformation matrix.
    """

    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv
