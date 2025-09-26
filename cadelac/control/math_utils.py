import numpy as np
from scipy.spatial.transform import Rotation


def rotation_z(theta):
	return np.array([[np.cos(theta), -np.sin(theta), 0],
			[np.sin(theta), np.cos(theta), 0],
			[0, 0, 1]])

# Define the rotation matrix around the x-axis
def rotation_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def rotation_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def skew_mat(x):
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])

def cross3(left,right):
	"""Numpy is inefficient for this"""
	return np.array([left[1] * right[2] - left[2] * right[1],
					 left[2] * right[0] - left[0] * right[2],
					 left[0] * right[1] - left[1] * right[0]])