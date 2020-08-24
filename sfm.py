from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import os
import pickle

from camera import Camera
from util import superpoint_match_two_way

IMAGES_DIR = "desk"
SUPERPOINT_DIR = "desk_superpoint"
MATCH_THRESHOLD = 0.7
FOCAL_LENGTH = 4308  # focal length in pixels

SUPERPOINT_HEIGHT = 120
SUPERPOINT_WIDTH = 160


def load_images_and_features(image_dir_filepath, superpoint_dir_filepath):
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
    matches_image = draw_matches(
        image_1, image_2, image_keypoints_1, image_keypoints_2, matches
    )
    cv2.imwrite(filename, matches_image)


def get_matches(images, keypoints, descriptors):
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
            matches[(index_1, index_2)] = superpoint_match_two_way(
                image_descriptors_1, image_descriptors_2, MATCH_THRESHOLD
            )
    return matches


def get_camera_poses(matches, keypoints, image_shape):
    """

    Args:
        matches (Dict[Tuple[int, int], ndarray]):
        keypoints:
        image_shape (Tuple[int, int]):

    Returns:
        (Dict[int, Dict[int, Dict[str, ndarray]]])
    """
    image_height, image_width = image_shape
    poses = defaultdict(dict)
    for indices, matches in matches.items():
        index_1, index_2 = indices
        pt1_matches = []
        pt2_matches = []
        for match_index in range(matches.shape[1]):
            pt1 = [
                int(keypoints[index_1][1][int(matches[0][match_index])]),
                int(keypoints[index_1][0][int(matches[0][match_index])]),
            ]
            pt2 = [
                int(keypoints[index_2][1][int(matches[1][match_index])]),
                int(keypoints[index_2][0][int(matches[1][match_index])]),
            ]
            pt1[0] = int(pt1[0] * image_height / SUPERPOINT_HEIGHT)
            pt2[0] = int(pt2[0] * image_height / SUPERPOINT_HEIGHT)

            pt1[1] = int(pt1[1] * image_width / SUPERPOINT_WIDTH)
            pt2[1] = int(pt2[1] * image_width / SUPERPOINT_WIDTH)
            pt1_matches.append((pt1[1], pt1[0]))
            pt2_matches.append((pt2[1], pt2[0]))
        e_mat, _ = cv2.findEssentialMat(
            np.array(pt1_matches), np.array(pt2_matches), focal=FOCAL_LENGTH
        )
        _, r_mat, t_vec, _ = cv2.recoverPose(
            e_mat, np.array(pt1_matches), np.array(pt2_matches), focal=FOCAL_LENGTH
        )
        poses[index_1][index_2] = {
            "rotation_matrix": r_mat,
            "translation_matrix": t_vec,
            "essential_matrix": e_mat,
        }

        e_mat, _ = cv2.findEssentialMat(
            np.array(pt2_matches), np.array(pt1_matches), focal=FOCAL_LENGTH
        )
        _, r_mat, t_vec, _ = cv2.recoverPose(
            e_mat, np.array(pt2_matches), np.array(pt1_matches), focal=FOCAL_LENGTH
        )
        poses[index_2][index_1] = {
            "rotation_matrix": r_mat,
            "translation_vector": t_vec,
            "essential_matrix": e_mat,
        }

    return poses


def populate_cameras(matches, keypoints, image_shape):
    poses = get_camera_poses(matches, keypoints, image_shape)
    cameras = {}
    for cam_index, pose in poses.items():
        rotations_translations = {
            ref_cam: (relation["rotation_matrix"], relation["translation_vector"])
            for ref_cam, relation in pose.items()
        }
        cameras[cam_index] = Camera(
            FOCAL_LENGTH, rotations_translations=rotations_translations
        )

    return cameras


def main():
    images, keypoints, descriptors = load_images_and_features(
        IMAGES_DIR, SUPERPOINT_DIR
    )
    matches = get_matches(images, keypoints, descriptors)
    image_shape = images[0].shape[:2]
    cameras = populate_cameras(matches, keypoints, image_shape)


if __name__ == "__main__":
    main()
