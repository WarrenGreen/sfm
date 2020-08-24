import numpy as np
from scipy.sparse import lil_matrix

from camera import Camera


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def projection_residuals(
    params, n_cameras, n_3d_points, camera_point_mappings, points_2d
):
    """
    Compute projection residuals.

    Args:
        params:
        n_cameras:
        n_3d_points:
        camera_point_mappings (List[Tuple[int, int, int]): List of camera param index, 3D point index and 2D point index mappings
        points_2d:

    Returns:

    """
    camera_params = params[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9 :].reshape((n_3d_points, 3))

    cameras = [Camera.from_vector_format(vec) for vec in camera_params]
    residuals = np.array([0, 0])
    for camera_index, point_3d_index, point_2d_index in camera_point_mappings:
        projected_point = cameras[camera_index].project(points_3d[point_3d_index])
        residuals += (projected_point - points_2d[point_2d_index]).ravel()

    return residuals
