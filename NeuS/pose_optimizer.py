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
from scipy.spatial.transform import Rotation as R


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
    rgba_img = cv2.imread(rgb_image_path,-1)
    rgb_img = rgba_img[...,:3]
    depth_img = cv2.imread(depth_image_path,-1)/1000

    camera_config_path = os.path.join(dataset_path, 'camera_info.json')
    camera_config = get_configs(camera_config_path)

    camera_intrinsic = np.array(camera_config[str(frame_idx)]["cam_intrinsic"])

    R_co = np.array(camera_config[str(frame_idx)]["cam_R_m2c"])
    t_co = np.array(camera_config[str(frame_idx)]["cam_t_m2c"])
    T_co = np.eye(4)
    T_co[:3,:3] = R_co
    T_co[:3,3] = t_co
    T_oc = np.linalg.inv(T_co)

    return rgb_img, depth_img, T_co, T_oc, camera_intrinsic

def save_results(Trans_mat_list):

    # for index, view in enumerate(Trans_mat_list):
    with open("./optimization_result.json", "w") as outfile: 
        json.dump(Trans_mat_list, outfile,indent=1)

def optimization_loop(optimizer,pts,T_init):
    return optimizer.estimate_pose_cam_obj(T_init,scale=1.0,pts=pts) # temp fix

if __name__ == "__main__":
    obj_name = 'owl'
    ckpt_path = './exp/neus/'+ obj_name +'/checkpoints/ckpt_005000.pth'
    sdf_network = load_checkpoint(ckpt_path)
    
    config_path = './confs/optimizer.json' # into args
    configs = get_configs(config_path)

    dataset_path = './data/pose_estimation/'
    frame_idx = 16
    rgb_img, depth_img, T_co, T_oc, camera_intrinsic = read_dataset_single_img(dataset_path,frame_idx)

    pts,_ = backproject(depth_img,camera_intrinsic, depth_img>0, NOCS_convention=False) # (16768, 3)

    # better downsample
    selected_indices = np.random.choice(pts.shape[0], 1000, replace=False) 
    pts = pts[selected_indices]
    mesh_path = os.path.join(dataset_path, 'owl.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    optimizer = Optimizer(sdf_network,configs)

    optimization_results = {}
    # batch process??
    for id in range(2):
        T_co_noised = T_co.copy()
        T_co_noised[:3, 3] =  T_co_noised[:3, 3] + np.random.rand(3)*0.3
        R_noised = R.from_euler('xyz', np.random.rand(3)*1, degrees=False)
        T_co_noised[:3, :3] =  T_co_noised[:3, :3] @ R_noised.as_matrix()

        T_co_optimized = optimization_loop(optimizer,pts,T_co_noised)
        T_co_optimized = T_co_optimized.detach().numpy()
        
        # get the inverse of the transformation
        T_oc_noised = np.linalg.inv(T_co_noised)
        T_co_optimized = np.linalg.inv(T_co_optimized)

        optimization_results[id] = {'T_oc_optimized': T_co_optimized.tolist(),
                                   'T_oc_noised': T_oc_noised.tolist(),
                                   'T_oc_gt': T_oc.tolist()
                                    }
        
        pts_in_obj_gt = pts@ T_oc[:3,:3].T+ T_oc[:3,3]
        pts_in_obj_noised = pts@ T_oc_noised[:3,:3].T + T_oc_noised[:3,3]
        pts_in_obj_optimized = pts@ T_co_optimized[:3,:3].T + T_co_optimized[:3,3]
        
        pcd_gt = visualize_points(pts_in_obj_gt)
        pcd_optimized = visualize_points(pts_in_obj_optimized)
        pcd_noised = visualize_points(pts_in_obj_noised)

        pcd_gt.paint_uniform_color([0, 1, 0]) # in green
        pcd_optimized.paint_uniform_color([0, 0, 1]) # in blue
        pcd_noised.paint_uniform_color([1, 0, 0]) # in red

        # o3d.visualization.draw_geometries([mesh,pcd_noised])
        # o3d.visualization.draw_geometries([mesh,pcd_optimized])
        o3d.visualization.draw([pcd_gt,pcd_optimized,pcd_noised, mesh])
        print(f'peak memory used after iteration {id} {torch.cuda.max_memory_allocated()*1e-9} GB')
    
    save_results(optimization_results)

    print("optimization done")
