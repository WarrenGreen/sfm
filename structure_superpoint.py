import pickle
import os

sp_dir = "desk_superpoint"

keypoints = []
for filename in os.listdir(sp_dir):
    pts, descs = pickle.load(open(os.path.join(sp_dir, filename), 'rb'))
    for index in range(descs.shape[1]):
        pt = pts[0][index], pts[1][index]
        desc = descs[index]
        kp = pt, desc
        keypoints.append(kp)
    pickle.dump(keypoints, open(os.path.join('desk_superpoint_structured/', filename), 'wb'))