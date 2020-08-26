import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from sfm.camera import Camera


def jac_sparsity_matrix(n_cameras, n_points, camera_indices, point_indices):
    """

    Args:
        n_cameras (int):
        n_points (int):
        camera_indices (ndarray):
        point_indices (ndarray):

    Returns:
        (ndarray)
    """
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    jac_mat = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        jac_mat[2 * i, camera_indices * 9 + s] = 1
        jac_mat[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        jac_mat[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        jac_mat[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return jac_mat


def projection_residuals(
    params, n_cameras, n_3d_points, camera_point_mappings, points_2d
):
    """
    Compute projection residuals.

    Args:
        params:
        n_cameras (int):
        n_3d_points (int):
        camera_point_mappings (List[Tuple[int, int, int]]):
        points_2d (ndarray):

    Returns:
        (ndarray)
    """
    camera_params = params[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9 :].reshape((n_3d_points, 3))

    cameras = [Camera.from_vector_format(vec) for vec in camera_params]
    residuals = np.array([0, 0])
    for camera_index, point_3d_index, point_2d_index in camera_point_mappings:
        projected_point = cameras[camera_index].project(points_3d[point_3d_index])
        residuals += (projected_point - points_2d[point_2d_index])

    return residuals


def optimize(residual_func, n_cameras, n_3d_points, camera_indices, points_3d_indices, points_2d):
    """

    Args:
        residual_func (Callable):
        n_cameras (int):
        n_3d_points (int):
        camera_point_mappings (List[Tuple[int, int, int]]):
        points_2d (ndarray):

    """
    x0 = residual_func()
    camera_indices = np.array(camera_indices)
    point_3d_indices = np.array(points_3d_indices)
    jac_sparsity_mat = jac_sparsity_matrix(
        n_cameras, n_3d_points, camera_indices, point_3d_indices
    )

    start_time = time.time()
    res = least_squares(
        residual_func,
        x0,
        jac_sparsity=jac_sparsity_mat,
        verbose=2,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        args=(n_cameras, n_3d_points, camera_indices, point_3d_indices, points_2d),
    )
    end_time = time.time()

    print(f"Optimization finished in {round(end_time - start_time)} seconds.")
