from collections import defaultdict
from pathlib import Path
from random import randint

import numpy as np
import cv2
import os
import pickle

from sfm.bundle_adjustment import optimize, projection_residuals
from sfm.camera import Camera
from sfm.util import superpoint_match_two_way

IMAGES_DIR = "data/desk/raw"
SUPERPOINT_DIR = "data/desk/superpoint"
MATCH_THRESHOLD = 0.7
FOCAL_LENGTH = 4308  # focal length in pixels

SUPERPOINT_HEIGHT = 120
SUPERPOINT_WIDTH = 160


def load_images_and_features(image_dir_filepath, superpoint_dir_filepath):
    """

    Args:
        image_dir_filepath:
        superpoint_dir_filepath:

    Returns:

    """
    images = []
    keypoints = []
    descriptors = []
    for filename in os.listdir(image_dir_filepath):
        if filename == ".DS_Store":
            # ignore system files
            continue
        img = cv2.imread(str(Path(image_dir_filepath) / filename))

        pts, descs = pickle.load(
            open(str(Path(superpoint_dir_filepath) / f"{filename[:-4]}.p"), "rb")
        )

        if len(pts) >= 2:
            images.append(img)
            keypoints.append(pts)
            descriptors.append(descs)
        else:
            print(
                f"Warning: {filename} did not have at least two detected keypoints "
                f"and will be ignored."
            )

    return images, keypoints, descriptors


def draw_matches(
    image_1,
    image_2,
    image_keypoints_1,
    image_keypoints_2,
    matches,
    stroke=3,
    color=(255, 0, 0),
):
    """

    Args:
        image_1:
        image_2:
        image_keypoints_1:
        image_keypoints_2:
        matches:
        stroke:
        color:

    Returns:

    """
    img1 = image_1.copy()
    o_h, o_w, _ = img1.shape
    img2 = image_2.copy()
    matches_image = np.hstack((img1, img2))
    pt1_matches = []
    pt2_matches = []

    for match_index in range(matches.shape[1]):
        pt1 = [
            int(image_keypoints_1[1][int(matches[0][match_index])]),
            int(image_keypoints_1[0][int(matches[0][match_index])]),
        ]
        pt2 = [
            int(image_keypoints_2[1][int(matches[1][match_index])]),
            int(image_keypoints_2[-1][0][int(matches[1][match_index])]),
        ]
        pt1[0] = int(pt1[0] * o_h / 120)
        pt2[0] = int(pt2[0] * o_h / 120)

        pt1[1] = int(pt1[1] * o_w / 160)
        pt2[1] = int(pt2[1] * o_w / 160 + o_w)
        pt1_matches.append((pt1[1], pt1[0]))
        pt2_matches.append((pt2[1], pt2[0]))
        cv2.line(
            matches_image,
            (pt1[1], pt1[0]),
            (pt2[1], pt2[0]),
            color=color,
            thickness=stroke,
            lineType=16,
        )

    return matches_image


def write_matches(
    image_1, image_2, image_keypoints_1, image_keypoints_2, matches, filename
):
    """

    Args:
        image_1:
        image_2:
        image_keypoints_1:
        image_keypoints_2:
        matches:
        filename:

    Returns:
        Writes image keypoint matches to filename.
    """
    matches_image = draw_matches(
        image_1, image_2, image_keypoints_1, image_keypoints_2, matches
    )
    cv2.imwrite(filename, matches_image)


def get_matches(images, keypoints, descriptors):
    """

    Args:
        images:
        keypoints:
        descriptors:

    Returns:
        (Dict[Tuple[int, int], ndarray])
    """
    matches = {}
    for index_1 in range(len(images)):
        for index_2 in range(len(images)):
            if index_1 == index_2:
                continue
            image_1, image_keypoints_1, image_descriptors_1 = (
                images[index_1],
                keypoints[index_1],
                descriptors[index_1],
            )
            image_2, image_keypoints_2, image_descriptors_2 = (
                images[index_2],
                keypoints[index_2],
                descriptors[index_2],
            )
            index_1_indices, index_2_indices, _ = superpoint_match_two_way(
                image_descriptors_1, image_descriptors_2, MATCH_THRESHOLD
            )
            index_1_indices = index_1_indices.astype(np.int)
            index_2_indices = index_2_indices.astype(np.int)
            matches[(index_1, index_2)] = keypoints[index_1].T[index_1_indices][:, :2].astype(np.int), keypoints[index_2].T[index_2_indices][:, :2].astype(np.int)
            matches[(index_2, index_1)] = keypoints[index_1].T[index_1_indices][:, :2].astype(np.int), keypoints[index_2].T[index_2_indices][:, :2].astype(np.int)
    return matches


def get_match_points(matches, keypoints, image_shape):
    """

    Args:
        matches:
        keypoints:
        image_shape:

    Returns:
        (Dict[Tuple[int, int], List[Tuple[int, int]]]):
    """
    image_height, image_width = image_shape
    match_points = defaultdict(list)
    for indices, points in matches.items():
        index_1, index_2 = indices
        image_1_points, image_2_points = points
        image_1_points[:, 0] = image_1_points[:, 0] * image_height / SUPERPOINT_HEIGHT
        image_2_points[:, 0] = image_2_points[:, 0] * image_height / SUPERPOINT_HEIGHT

        image_1_points[:, 1] = image_1_points[:, 1] * image_width / SUPERPOINT_WIDTH
        image_2_points[:, 1] = image_2_points[:, 1] * image_width / SUPERPOINT_WIDTH

        match_points[(index_1, index_2)] = np.hstack((image_1_points, image_2_points)).reshape((-1,2,2)).tolist()
        match_points[(index_2, index_1)] = np.hstack(
            (image_2_points, image_1_points)).reshape((-1, 2, 2)).tolist()
    return match_points


def get_camera_poses(match_points):
    """
    Compute camera poses given keypoint matches.

    Args:
        match_points (Dict[Tuple[int, int], Tuple[Tuple[int,int]]]):

    Returns:
        (Dict[int, Dict[int, Dict[str, ndarray]]]):
            Camera index -> reference camera index -> matrix values
    """
    poses = defaultdict(dict)
    for indices, matches in match_points.items():
        index_1, index_2 = indices
        cam1_matches = []
        cam2_matches = []
        for cam_1_point, cam_2_point in matches:
            cam1_matches.append(cam_1_point)
            cam2_matches.append(cam_2_point)

        e_mat, _ = cv2.findEssentialMat(
            np.array(cam1_matches), np.array(cam2_matches), focal=FOCAL_LENGTH
        )
        f_mat, _ = cv2.findFundamentalMat(np.array(cam1_matches), np.array(cam2_matches))
        _, r_mat, t_vec, _ = cv2.recoverPose(
            e_mat, np.array(cam1_matches), np.array(cam2_matches), focal=FOCAL_LENGTH
        )
        cam_1_points = np.array(cam1_matches)
        if f_mat is not None and f_mat.shape == (3,3):
            lines = cv2.computeCorrespondEpilines(cam_1_points, 1, f_mat)
        else:
            lines = None
        poses[index_1][index_2] = {
            "rotation_matrix": r_mat,
            "translation_vector": t_vec,
            "essential_matrix": e_mat,
            "fundamental_matrix": f_mat,
            "lines": lines
        }

        e_mat, _ = cv2.findEssentialMat(
            np.array(cam2_matches), np.array(cam1_matches), focal=FOCAL_LENGTH
        )
        f_mat, _ = cv2.findFundamentalMat(np.array(cam2_matches), np.array(cam1_matches))
        _, r_mat, t_vec, _ = cv2.recoverPose(
            e_mat, np.array(cam2_matches), np.array(cam1_matches), focal=FOCAL_LENGTH
        )
        if f_mat is not None and f_mat.shape == (3, 3):
            lines = cv2.computeCorrespondEpilines(cam_1_points, 1, f_mat)
        else:
            lines = None
        poses[index_2][index_1] = {
            "rotation_matrix": r_mat,
            "translation_vector": t_vec,
            "essential_matrix": e_mat,
            "fundamental_matrix": f_mat,
            "lines": lines
        }

    return poses


def populate_cameras(match_points):
    """
    Create camera objects with relative poses given keypoint matches.

    Args:
        match_points:

    Returns:
        (Dict[int, Camera):
    """
    poses = get_camera_poses(match_points)
    cameras = {}
    for cam_index, pose in poses.items():
        rotations_translations = {
            ref_cam: (relation["rotation_matrix"], relation["translation_vector"])
            for ref_cam, relation in pose.items()
        }
        cameras[cam_index] = Camera(
            FOCAL_LENGTH, rotations_translations=rotations_translations
        )
        cameras[cam_index].info = pose

    return cameras


def populate_3d_points(cameras, match_points):
    points_3d = []
    points_3d_indices = []
    points_2d = []
    cam_indices = []
    for indices, matches in match_points.items():
        index_1, index_2 = indices
        points_2d_1, points_2d_2 = [], []

        for cam_1_point, cam_2_point in matches:
            points_2d.append(cam_1_point)
            points_2d_1.append(cam_1_point)
            cam_indices.append(index_1)
            points_2d.append(cam_2_point)
            cam_indices.append(index_2)
            points_2d_2.append(cam_2_point)

        projection_matrix_1 = cameras[index_1].get_projection_matrix(None)
        projection_matrix_2 = cameras[index_2].get_projection_matrix(index_1)
        import pdb;pdb.set_trace()
        points_4d = cv2.triangulatePoints(
            projection_matrix_1,
            projection_matrix_2,
            np.array(points_2d_1),
            np.array(points_2d_2)
        )
        points_4d_list = points_4d.tolist()
        points_3d_indices += list(range(len(points_3d), len(points_3d) + len(points_4d_list)))
        points_3d += points_4d_list
    return points_2d, cam_indices, points_3d, points_3d_indices


def main():
    images, keypoints, descriptors = load_images_and_features(
        IMAGES_DIR, SUPERPOINT_DIR
    )
    matches = get_matches(images, keypoints, descriptors)
    image_shape = images[0].shape[:2]
    match_points = get_match_points(matches, keypoints, image_shape)
    cameras = populate_cameras(match_points)
    img_large_1 = images[0]
    img_large_2 = images[1]
    height, width = image_shape

    for index in range(len(cameras[1].info[0]['lines'])):
        line = cameras[1].info[0]['lines'][index]
        match_point = match_points[(1, 0)][index][0]
        aa, bb, cc = line[0]
        pt1 = 0, height - int(-cc/bb)
        pt2 = width, int(height-((-cc - aa*width)/bb))

        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        img_large_1 = cv2.line(img_large_1, pt2, pt1, color=color, thickness=20)
        img_large_2 = cv2.circle(img_large_2, (match_point[0], height-match_point[1]), radius=50,
                          color=color, thickness=-1)

    img1 = cv2.resize(img_large_1, (640, 480))
    img2 = cv2.resize(img_large_2, (640, 480))
    merged = np.hstack((img2, img1))
    cv2.imshow("img", merged)
    cv2.waitKey(0)

    cv2.imwrite("epipolar_lines_0_1.png", merged)

    points_2d, cam_indices, points_3d, points_3d_indices = populate_3d_points(cameras, match_points)

    optimize(
        residual_func=projection_residuals,
        n_cameras=len(cameras),
        n_3d_points=len(points_3d),
        camera_indices=cam_indices,
        points_3d_indices=points_3d_indices,
        points_2d=points_2d
    )


if __name__ == "__main__":
    main()
