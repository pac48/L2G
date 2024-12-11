import os
import time

from l2g_core.graspsamplenet import GraspSampleNet
from l2g_core.pytorch_utils import load_checkpoint
import open3d as o3d
import numpy as np
import torch
import glob

import onnxruntime as ort


def update(ort_sess, file):
    # Load the PCD file
    # cloud = pypcd.PointCloud.from_path("labsim_test.pcd")
    pcd = o3d.io.read_point_cloud(file)
    # Access the point cloud data as a numpy array
    points = np.array(pcd.points)
    if points.shape[0] > 10000:
        inds = np.linspace(0, points.shape[0] - 1, 10000, dtype=np.int32)
        points = points[inds, :]

    points = np.reshape(points, (1, -1, 3))
    points = points.astype(np.float32)
    predicted_grasps, predicted_scores = ort_sess.run(None, {'point_cloud': points})

    print(predicted_grasps)
    predicted_grasps = predicted_grasps.flatten()
    predicted_scores = predicted_scores
    predicted_grasps_bytes = predicted_grasps.tobytes()
    predicted_scores_bytes = predicted_scores.tobytes()

    with open(f"/tmp/{os.path.basename(file).replace('.pcd', '.bin')}", 'wb') as f:
        f.write(predicted_grasps_bytes)


def run():
    # MODEL DEFINITION
    ort_sess = ort.InferenceSession('l2g.onnx', providers=['CUDAExecutionProvider'])

    while True:
        time.sleep(0.1)
        files = glob.glob("/tmp/*.pcd")
        files = ['snapshot-20241211161050.pcd']
        for file in files:
            update(ort_sess, file)
            # os.remove(file)


if __name__ == "__main__":
    run()
