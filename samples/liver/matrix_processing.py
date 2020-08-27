import numpy as np
import math
import random
import warnings


# generate 4x4 rotation matrix
# axis : [x, y, z] angle : degree
def rotation_around_axis(axis, angle):
    warnings.filterwarnings("ignore")
    angle = np.radians(angle)
    axis = axis / np.linalg.norm(axis)

    a = math.cos(angle / 2.0)
    b, c, d = axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array([[aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac), 0],
                     [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab), 0],
                     [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


# generate 4x4 translation matrix
def translation(vector):
    return np.array([[1, 0, 0, vector[0]],
                     [0, 1, 0, vector[1]],
                     [0, 0, 1, vector[2]],
                     [0, 0, 0, 1]])


def _cot(x):
    return math.cos(x) / math.sin(x)


def shear_along_axis(axis_num, angle):
    if axis_num == 0:
        return shear_along_x(angle)
    elif axis_num == 1:
        return shear_along_y(angle)
    elif axis_num == 2:
        return shear_along_z(angle)


# angle : degree
def shear_along_x(angle):
    angle = np.radians(90 - angle)
    return np.array([[1, _cot(angle), 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


# angle : degree
def shear_along_y(angle):
    angle = np.radians(90 - angle)
    return np.array([[1, 0, 0, 0],
                     [_cot(angle), 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


# angle : degree
def shear_along_z(angle):
    angle = np.radians(90 - angle)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [_cot(angle), 0, 1, 0],
                     [0, 0, 0, 1]])


def _system_random_between(low, high):
    return low + (high - low) * random.SystemRandom().random()


# generate 4x4 random rotation matrix
# axis : [x, y, z] angle : degree
def generate_random_rotation_around_axis(axis, angle):
    warnings.filterwarnings("ignore")
    random_angle = _system_random_between(-abs(angle), abs(angle))
    # print(random_angle)
    return rotation_around_axis(axis, random_angle)


# generate 4x4 random shear matrix
# angle : degree
def generate_random_shear(angle):
    warnings.filterwarnings("ignore")
    random_angles = [_system_random_between(-abs(angle), abs(angle)) for i in range(3)]
    matrix = np.matmul(shear_along_axis(0, random_angles[0]), shear_along_axis(1, random_angles[1]))
    matrix = np.matmul(matrix, shear_along_axis(2, random_angles[2]))
    return matrix
