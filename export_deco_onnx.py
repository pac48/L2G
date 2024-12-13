from l2g_core.graspsamplenet import GraspSampleNet
import torch
import torch.nn as nn
import numpy as np
from l2g_core.utils.grasp_utils import reparametrize_grasps


class GraspSampleNetONNX(nn.Module):
    def __init__(self):
        super().__init__()
        feat = "deco"
        deco_config = 'deco/deco_config.yaml'
        sampled_grasps = 500
        sample_group_size = 10  # maybe?
        neigh_size = 100
        train_temperature = True  # maybe?
        use_angle_feat = True
        neigh_aggr = "w_avg"  # maybe?
        self.model = GraspSampleNet(
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
            resume=False
        )
        self.model = self.model.cuda()

        path = "checkpoints_L2G/deco_l2g_nn100_grasps500/opt-adam_lr0.0001_lr-step100_wd0.0001_epochs500_seed21996/checkpoints/epoch_500.pth"
        checkpoint = torch.load(path)
        res_load_weights = self.model.load_state_dict(checkpoint['model'], strict=True)

        # set eval mode
        self.model = self.model.eval()

    def forward(self, points_tensor):
        predicted_grasps, predicted_scores = self.model(points_tensor, gt_sampling=None, gt_grasps=None)
        return reparametrize_grasps(predicted_grasps, with_width=True, gpnet_scale=True)


def run():
    # MODEL DEFINITION

    model = GraspSampleNetONNX()
    points_tensor = torch.tensor(np.random.rand(1, 5000, 3)).float().cuda()
    torch.onnx.export(
        model,  # model to export
        (points_tensor,),  # inputs of the model,
        "l2g.onnx",  # filename of the ONNX model
        input_names=["point_cloud"],  # Rename inputs for the ONNX model
        output_names=["predicted_grasps"],  # Rename inputs for the ONNX model
        dynamic_axes={'point_cloud': {1: 'num_points'},
                      'predicted_grasps': {1: 'num_grasps'},
                      'grasp_scores': {1: 'num_grasps'}},  # variable length axes
        opset_version=13,
    )


if __name__ == "__main__":
    run()
