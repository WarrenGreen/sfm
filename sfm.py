import numpy as np
import cv2
import os
import pickle

from util import nn_match_two_way

images_dir = "desk/"
MATCH_THRESHOLD = 0.7
FOCAL_LENGTH = 4308 # focal length in pixels
stroke = 3
color = (255,0,0)


images = []
gray_images = []
keypoints = []
descriptors = []

for filename in os.listdir(images_dir):
    if filename == '.DS_Store':
        continue
    img = cv2.imread(os.path.join(images_dir, filename))
    images.append(img)

    pts, descs = pickle.load(open('desk_superpoint/{}.p'.format(filename[:-4]), 'rb'))

    keypoints.append(pts)
    descriptors.append(descs)
    if len(keypoints) < 2:
        continue

    matches = nn_match_two_way(descriptors[-2], descriptors[-1], MATCH_THRESHOLD)

    img1 = images[-2].copy()
    o_h, o_w, _ = img1.shape
    img2 = images[-1].copy()
    img3 = np.hstack((img1, img2))
    pt1_matches = []
    pt2_matches = []

    for match_index in range(matches.shape[1]):
        pt1 = [int(keypoints[-2][1][int(matches[0][match_index])]), int(keypoints[-2][0][int(matches[0][match_index])])]
        pt2 = [int(keypoints[-1][1][int(matches[1][match_index])]),  int(keypoints[-1][0][int(matches[1][match_index])])]
        pt1[0] = int(pt1[0] * o_h / 120)
        pt2[0] = int(pt2[0] * o_h / 120)

        pt1[1] = int(pt1[1] * o_w / 160)
        pt2[1] = int(pt2[1] * o_w / 160 + o_w)
        pt1_matches.append((pt1[1], pt1[0]),)
        pt2_matches.append((pt2[1], pt2[0]),)
        cv2.line(img3, (pt1[1], pt1[0]), (pt2[1], pt2[0]), color=color, thickness=stroke, lineType=16)


    cv2.imwrite(os.path.join('desk_point_matches', filename), img3)

    e_mat, _ = cv2.findEssentialMat(np.array(pt1_matches), np.array(pt2_matches), focal=FOCAL_LENGTH)
    _, r_mat, t_vec, _ = cv2.recoverPose(e_mat, np.array(pt1_matches), np.array(pt2_matches), focal=FOCAL_LENGTH)
    print(r_mat, t_vec)

    
