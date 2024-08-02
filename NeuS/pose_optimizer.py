import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import torch.nn as nn
import math
import os
import cv2
from PIL import Image
import ipdb
import json
from pyhocon import ConfigFactory

from models.fields import FeatureField, SDFNetwork
from reconstruct.optimizer import Optimizer
from reconstruct.utils import color_table, set_view, get_configs
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(checkpoint_fname):
    # read network config
    conf_path = './confs/wmask.conf'
    f = open(conf_path)
    conf_text = f.read()
    conf_text = conf_text.replace('CASE_NAME', 'owl') #TODO case name as input
    f.close()
    conf = ConfigFactory.parse_string(conf_text)

    # load feature field
    checkpoint = torch.load(os.path.join(checkpoint_fname), map_location='cuda')
    sdf_network = SDFNetwork(**conf['model.sdf_network']).to(device)
    # feature_network = FeatureField(**conf['model.feature_field']).to('cuda')
    # self.nerf_outside.load_state_dict(checkpoint['nerf'])
    sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
    # self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
    # self.color_network.load_state_dict(checkpoint['color_network_fine'])
    # feature_network.load_state_dict(checkpoint['feature_network'])
    # self.optimizer_geometry.load_state_dict(checkpoint['optimizer-geometry'])
    # self.optimizer_feature.load_state_dict(checkpoint['optimizer-feature'])

    # iter_step = checkpoint['iter_step']
    print(f"loaded checkpoint from {checkpoint_fname}")
    return sdf_network


# helper function for pointcloud
def visualize_points(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_point_cloud(pcd):
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud"""

    def __init__(self, height, width):
        super(BackprojectDepth, self).__init__()

        # self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False
        )
        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0
        )

    def forward(self, depth, K):
        if isinstance(K, np.ndarray):
            assert K.shape == (3, 3)
            K = torch.from_numpy(K).float().to(depth.device)[None]

        batch_size = depth.shape[0]
        ones = torch.ones(batch_size, 1, self.height * self.width).to(depth.device)
        inv_K = torch.inverse(K).to(depth.device)  # [B, 3, 3]

        pix_coords = self.pix_coords.clone().to(depth.device)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        pix_coords = torch.cat([pix_coords, ones], 1)  # [B, 3, H*W]

        cam_points = torch.matmul(inv_K, pix_coords)  # [B, 3, 3] @ [B, 3, H*W]
        cam_points = (
            depth.view(batch_size, 1, -1) * cam_points
        )  # [B, 1, H*W] * [B, 3, H*W]
        return cam_points

def backproject(depth, intrinsics, instance_mask, NOCS_convention=True):
    intrinsics_inv = np.linalg.inv(intrinsics)
    # image_shape = depth.shape
    # width = image_shape[1]
    # height = image_shape[0]

    # x = np.arange(width)
    # y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixsel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    if NOCS_convention:
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]
    return pts, idxs

def read_dataset_single_img(dataset_path, frame_idx):
    # image file name
    rgb_image_path = os.path.join(dataset_path, 'rgbsyn_init_calib/{:06d}.png'.format(frame_idx))
    depth_image_path = os.path.join(dataset_path, 'depth_init_calib/{:06d}.png'.format(frame_idx))

    # read image
    rgb_img =  Image.open(rgb_image_path)
    depth_img = Image.open(depth_image_path)
    # depth_img = cv2.imread(depth_image_path).astype(np.uint16)

    camera_config_path = os.path.join(dataset_path, 'camera_info.json')
    camera_config = get_configs(camera_config_path)

    camera_intrinsic = np.array(camera_config[str(frame_idx)]["cam_intrinsic"])

    R_o2c = np.array(camera_config[str(frame_idx)]["cam_R_m2c"])
    t_o2c = np.array(camera_config[str(frame_idx)]["cam_t_m2c"])
    T_o2c = np.eye(4)
    T_o2c[:3,:3] = R_o2c
    T_o2c[:3,3] = t_o2c
    # T_c2o
    T_c2o = np.eye(4)
    T_c2o[:3,:3] = R_o2c.T
    T_c2o[:3,3] = -t_o2c.dot(R_o2c.T) 

    return rgb_img, depth_img, T_o2c, T_c2o, camera_intrinsic

def save_results(Trans_mat_list):

    # for index, view in enumerate(Trans_mat_list):
    with open("./optimization_result.json", "w") as outfile: 
        json.dump(Trans_mat_list, outfile,indent=1)

def optimization_loop(optimizer,pts,T_init):
   
    return optimizer.estimate_pose_cam_obj(T_init,scale=1.0,pts=pts[:32768,:]) # temp fix

if __name__ == "__main__":
    obj_name = 'owl'
    ckpt_path = './exp/neus/'+ obj_name +'/checkpoints/ckpt_005000.pth'
    print(ckpt_path)
    config_path = './confs/optimizer.json' # into args
    configs = get_configs(config_path)

    sdf_network = load_checkpoint(ckpt_path)

    dataset_path = './data/pose_estimation/'
    frame_idx = 16
    rgb_img, depth_img, T_o2c, T_c2o, camera_intrinsic = read_dataset_single_img(dataset_path,frame_idx)

    transform = transforms.ToTensor()
    depth_tensor = transform(depth_img)
    back_prop = BackprojectDepth(height=depth_img.height,width=depth_img.width)

    pts = back_prop(depth_tensor,camera_intrinsic) # (1, 3, 65536)
    pts = torch.squeeze(pts.permute(0,2,1),dim=0) # (65536, 3)
    pts = pts/1000
    pts = pts.detach().numpy()
   
    optimizer = Optimizer(sdf_network,configs)

    
    
    T_optimized = {}
    for id in range(2):
        T_c2o_noised = T_c2o.copy()
        T_c2o_noised[:3, 3] =  T_c2o_noised[:3, 3] + np.random.rand(3)*0.01 
        T_c2o_optimized = optimization_loop(optimizer,pts,T_c2o_noised)
        print(T_c2o_optimized)
        T_optimized[id] = {'T_c2o_optimized': T_c2o_optimized.tolist()}
    save_results(T_c2o_optimized)