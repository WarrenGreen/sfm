from _ast import Tuple

import cv2
import numpy as np

from sfm.util import get_int_or_tuple


class Camera:
    def __init__(self, focal_length, skew=0, center=0, rotations_translations=None):
        """

        Args:
            focal_length (Union[int, Tuple[int, int]]): focal length in pixels.
                Either a single int to be used for both x and y values or a tuple
                containing (f_x, f_y).
            skew (int): intrinsic skew
            center (Union[int, Tuple[int, int]]): principle point in pixels.
                Either a single int to be used for both x and y values or a tuple
                containing (c_x, c_y).
            rotations_translations (Optional[Dict[int, Tuple[ndarray, ndarray]]]): Map
                of rotation matrices and translation vectors with respect to cameras
                referenced by camera index.
        """
        self.focal_x, self.focal_y = get_int_or_tuple(focal_length)
        self.center_x, self.center_y = get_int_or_tuple(center)
        self.skew = skew

        self._rotation_matrices = {}
        self._rotation_vectors = {}
        self._translation_vectors = {}
        for ref_cam_index, extrinsics in rotations_translations.items():
            r_matrix, t_vec = extrinsics
            self._rotation_matrices[ref_cam_index] = r_matrix
            self._translation_vectors[ref_cam_index] = t_vec
            self._rotation_vectors[ref_cam_index] = cv2.Rodrigues(r_matrix)

        mat = np.zeros((3, 3))
        mat[0][0] = self.focal_x
        mat[1][1] = self.focal_y
        mat[0][1] = self.skew
        mat[0][2] = self.center_x
        mat[1][2] = self.center_y
        mat[2][2] = 1
        self.intrinsics_matrix = mat

    def get_extrinsics_matrix(self, ref_camera_index):
        """
        Get rotation matrix and translation vector in respect to a given camera.

        Args:
            ref_camera_index: reference camera for rotation/translation.

        Returns:
            (Tuple[ndarray, ndarray])
        """
        if ref_camera_index is None:
            r_mat = np.eye(3)
            t_vec = np.zeros((1, 3))
            return r_mat, t_vec
        else:
            return (
                self._rotation_matrices[ref_camera_index],
                self._translation_vectors[ref_camera_index],
            )

    def get_translation_vector(self, ref_camera_index):
        return self._translation_vectors[ref_camera_index]

    def get_rotation_vector(self, ref_camera_index):
        return self._rotation_vectors[ref_camera_index]

    def set_rotation_vector(self, ref_camera_index, rotation_vector):
        self._rotation_vectors[ref_camera_index] = rotation_vector
        self._rotation_matrices[ref_camera_index] = cv2.Rodrigues(rotation_vector)

    def get_rotation_matrix(self, ref_camera_index):
        return self._rotation_matrices[ref_camera_index]

    def set_rotation_matrix(self, ref_camera_index, rotation_matrix):
        self._rotation_matrices[ref_camera_index] = rotation_matrix
        self._rotation_vectors[ref_camera_index] = cv2.Rodrigues(rotation_matrix)

    def project(self, point_3d, ref_camera_index=0):
        """
        Project 3D point to 2D u,v coordinates in camera space.
        Args:
            point_3d (ndarray[int, int, int]): X,Y,Z
            ref_camera_index (int):

        Returns:
            (Tuple[int, int]): u,v
        """
        r_mat, t_vec = self.get_extrinsics_matrix(ref_camera_index)
        x, y, z = r_mat * point_3d + t_vec
        x_prime = x / z
        y_prime = y / z
        u = self.focal_x * x_prime + self.center_x
        v = self.focal_y * y_prime + self.center_y
        return u, v

    def get_projection_matrix(self, ref_camera_index):
        r_mat, t_vec = self.get_extrinsics_matrix(ref_camera_index)
        rt_mat = np.hstack((r_mat,  t_vec))
        k_mat = self.intrinsics_matrix
        return np.dot(k_mat, rt_mat)

    def to_vector_format(self, ref_camera_index):
        """ Vector format used for bundle adjustment. """
        return np.concatenate(
            self.get_rotation_vector(ref_camera_index),
            self.get_translation_vector(ref_camera_index),
            np.array([self.focal_x, self.focal_y]),
        )

    @classmethod
    def from_vector_format(cls, vector, ref_camera_index=0):
        """ Vector format used for bundle adjustment. """
        r_vec = vector[:3]
        r_mat = cv2.Rodrigues(r_vec)
        t_vec = vector[3:7]
        f_x, f_y = vector[7:]
        return cls(
            focal_length=(f_x, f_y),
            rotations_translations={ref_camera_index: (r_mat, t_vec)},
        )
