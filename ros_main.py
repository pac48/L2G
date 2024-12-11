import os
import time

from l2g_core.graspsamplenet import GraspSampleNet
from l2g_core.pytorch_utils import load_checkpoint
import open3d as o3d
import numpy as np
import torch
import glob


def update(model, file):
    # Load the PCD file
    # cloud = pypcd.PointCloud.from_path("labsim_test.pcd")
    pcd = o3d.io.read_point_cloud(file)
    # Access the point cloud data as a numpy array
    points = np.array(pcd.points)
    if points.shape[0] > 10000:
        inds = np.linspace(0, points.shape[0] - 1, 10000, dtype=np.int32)
        points = points[inds, :]

    points_tensor = torch.tensor(points).float().cuda()
    points_tensor = points_tensor.unsqueeze(0)
    predicted_grasps, predicted_scores = model(points_tensor, gt_sampling=None, gt_grasps=None)

    print(predicted_grasps)
    predicted_grasps = predicted_grasps.cpu().detach().numpy().flatten()
    predicted_scores = predicted_scores.cpu().detach().numpy()
    predicted_grasps_bytes = predicted_grasps.tobytes()
    predicted_scores_bytes = predicted_scores.tobytes()

    with open(f"/tmp/{os.path.basename(file).replace('.pcd', '.bin')}", 'wb') as f:
        f.write(predicted_grasps_bytes)


def run():
    # MODEL DEFINITION
    feat = "deco"
    deco_config = 'deco/deco_config.yaml'
    sampled_grasps = 500
    sample_group_size = 10  # maybe?
    neigh_size = 100
    train_temperature = True  # maybe?
    use_angle_feat = True
    neigh_aggr = "w_avg"  # maybe?
    model = GraspSampleNet(
        feat_extractor=feat,
        deco_config_path=deco_config,
        sampled_grasps=sampled_grasps,
        sample_group_size=sample_group_size,
        simp_loss='chamfer',
        train_temperature=train_temperature,
        neigh_size=neigh_size,
        use_all_grasp_info=False,
        use_contact_angle_feat=use_angle_feat,
        angle_feat_depth=2,
        projected_feat_aggregation=neigh_aggr,
        bn=False,
        resume=True
    )
    model = model.cuda()
    path = "checkpoints_L2G/deco_l2g_nn100_grasps500/opt-adam_lr0.0001_lr-step100_wd0.0001_epochs500_seed21996/checkpoints/epoch_500.pth"
    checkpoint = torch.load(path)
    res_load_weights = model.load_state_dict(checkpoint['model'], strict=True)

    # set eval mode
    model = model.eval()

    while True:
        time.sleep(0.1)
        files = glob.glob("/tmp/*.pcd")
        files = ['snapshot-20241211161050.pcd']
        for file in files:
            update(model, file)
            # os.remove(file)


if __name__ == "__main__":
    run()
