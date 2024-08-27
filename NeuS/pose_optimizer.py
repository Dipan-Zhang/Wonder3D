import torch
import numpy as np
import open3d as o3d
import os
import cv2
import json
import argparse
from pyhocon import ConfigFactory

from models.fields import FeatureField, SDFNetwork
from reconstruct.optimizer import Optimizer
from reconstruct.utils import color_table, set_view, get_configs
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(checkpoint_fname):
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


def save_dict(save_path, dict):
    """a dict to a json file"""
    with open(save_path, "w") as outfile: 
        json.dump(dict, outfile,indent=1)

def Rz_rand():
    """create a random rotation matrix around z axis"""
    theta = np.random.rand()*2*np.pi
    R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    return R_z

def Ry_rand():
    """create a random rotation matrix around y axis"""
    theta = np.random.rand()*2*np.pi
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return R_y

def Rx_rand():
    """create a random rotation matrix around x axis"""
    theta = np.random.rand()*2*np.pi
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return R_x

def Rx_rand_limited():
    """create a random rotation matrix around x axis with range 0-PI"""
    theta = np.random.rand()*np.pi
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return R_x

def sample_T_oc_init_list(n,translation):
    T_list = []
    for _ in range(n):
        T_co_init = np.eye(4)
        T_co_init[:3, :3] = Rx_rand_limited()
        T_co_init[:3, 3] = translation 
        T_list.append(T_co_init)
    return T_list


def rank_T_init(sdf_network,pts,T_init_list):
    """rank the sdf value of the points based on the rotation matrix"""
    sdf_list = []
    for T_init in T_init_list:
        pts_in_obj = pts@ T_init[:3,:3].T + T_init[:3,3]
        sdf = sdf_network.sdf(torch.tensor(pts_in_obj).to(device).float())
        sdf_list.append(sdf.mean())
    sdf_list = torch.stack(sdf_list)
    min_idx =  torch.argmin(torch.abs(sdf_list))

    return T_init_list[min_idx],min_idx


def owl():
    """case for rendered dataset"""
    obj_name = 'owl'
    ckpt_path = './exp/neus/'+ obj_name +'/checkpoints/ckpt_005000.pth'
    sdf_network = load_checkpoint(ckpt_path)
    mesh_path = os.path.join(dataset_path, 'owl.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    dataset_path = '../customized_data/owl_rendered/pose_estimation/'
    frame_idx = 16
    rgb_img, depth_img, T_co, T_oc, camera_intrinsic = read_dataset_single_img(dataset_path,frame_idx)

    pts,_ = backproject(depth_img,camera_intrinsic, depth_img>0, NOCS_convention=False) # (16768, 3)

    selected_indices = np.random.choice(pts.shape[0], 1000, replace=False) 
    pts = pts[selected_indices]

    # estimate scale
    bounding_box_mesh = mesh.get_axis_aligned_bounding_box()
    diagonal_length_mesh = np.linalg.norm(bounding_box_mesh.get_max_bound() - bounding_box_mesh.get_min_bound())
    diagonal_length_pts = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    scale = diagonal_length_mesh/diagonal_length_pts
    print(f'Estimate initial scale {scale}')

    translation_guess = pts.mean(axis=0)
    T_co_init_list = sample_T_oc_init_list(10,translation_guess)
    T_co_init_ranked, _ = rank_T_init(sdf_network,pts,T_co_init_list)

    return pts, mesh, sdf_network, T_co, T_co_init_ranked,scale


def cup_hz():
    obj_name = 'cup_hz'
    ckpt_path = './exp/neus/'+ obj_name +'/checkpoints/ckpt_003000.pth'
    sdf_network = load_checkpoint(ckpt_path)
    mesh_path = '/home/stud/zanr/code/tmp/Wonder3D/NeuS/exp/neus/cup_hz/meshes/cup_hz_3000.glb'
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # load dataset
    dataset_path = '../customized_data/cpu_hz/'
    rgb_image_path = os.path.join(dataset_path, 'object_rgba.png')
    mask_image_path = os.path.join(dataset_path, 'object_pose/0_mask.png')
    depth_image_path = os.path.join(dataset_path, 'depth/000001.png')

    rgba_image = cv2.imread(rgb_image_path, -1)
    rgb_image = rgba_image[..., :3][...,[2,1,0]]
    mask_image = cv2.imread(mask_image_path, -1)
    depth_img = cv2.imread(depth_image_path, -1)/1000 

    camera_intrinsic = np.array(
            [
                [608.7960205078125, 0, 632.1019287109375],
                [0, 608.9515380859375, 365.985595703125],
                [0, 0, 1],
            ]
        )
    
    # quasi gt
    T_co_gt = np.loadtxt(os.path.join(dataset_path, 'object_pose/0_Rt_cam_obj.txt')).reshape(4, 4)
    rot_x90 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    T_co_gt = T_co_gt @ rot_x90.T

    pts,_ = backproject(depth_img,camera_intrinsic, mask_image, NOCS_convention=False)

    selected_indices = np.random.choice(pts.shape[0], 1000, replace=False) 
    pts = pts[selected_indices]

    # estimate scale
    bounding_box_mesh = mesh.get_axis_aligned_bounding_box()
    diagonal_length_mesh = np.linalg.norm(bounding_box_mesh.get_max_bound() - bounding_box_mesh.get_min_bound())
    diagonal_length_pts = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    scale = diagonal_length_mesh/diagonal_length_pts
    print(f'Estimate initial scale {scale}')

    translation_guess = pts.mean(axis=0)
    T_co_init_list = sample_T_oc_init_list(10,translation_guess)
    T_co_init_ranked, _ = rank_T_init(sdf_network,pts,T_co_init_list)
    
    return pts, mesh, sdf_network, T_co_gt, T_co_init_ranked,scale


def load_kinect_dataset(case_name,mask_id=0):
    """load self recorded kinect dataset"""
    ckpt_path = './exp/neus/'+ case_name +'/checkpoints/ckpt_003000.pth'
    sdf_network = load_checkpoint(ckpt_path)

    mesh_path = f'/home/stud/zanr/code/tmp/Wonder3D/NeuS/exp/neus/{case_name}/meshes/{case_name}_3000.glb'
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    dataset_path = '../customized_data/table_top/'
    rgb_image_path = os.path.join(dataset_path, 'object_rgba_1.png')
    # depth_image_path = os.path.join(dataset_path, 'object_depth.png')
    mask_image_path = os.path.join(dataset_path, f'masks/mask_{mask_id}.png')
    depth_image_path = os.path.join(dataset_path, 'depth/00001.png')
    background_mask_path = os.path.join(dataset_path, 'desk_mask.png')
    # depth_image_path = '/home/stud/zanr/code/tmp/Wonder3D/customized_data/2023-12-05-17-44-48/object_depth_test.png'

    # read image
    rgba_image = cv2.imread(rgb_image_path, -1)
    rgb_image = rgba_image[..., :3][...,[2,1,0]]
    mask_image = cv2.imread(mask_image_path, -1)
    desk_mask = cv2.imread(background_mask_path, -1)

    depth_img = cv2.imread(depth_image_path, -1)/1000 
    print(f'The depth mean is {depth_img.mean()}')

    camera_intrinsic = np.array(
            [
                [608.7960205078125, 0, 632.1019287109375],
                [0, 608.9515380859375, 365.985595703125],
                [0, 0, 1],
            ]
        )
    pts, _ = backproject(depth_img, camera_intrinsic,mask_image, NOCS_convention=False)
    selected_indices = np.random.choice(pts.shape[0], 1000, replace=False) 
    pts = pts[selected_indices]

    pcd = visualize_points(pts)
    pcd_remove_outlier, _ = pcd.remove_radius_outlier(10,0.2)
    pcd_remove_outlier, _ = pcd_remove_outlier.remove_statistical_outlier(nb_neighbors=40,std_ratio=2.0)
    pts_removed = np.asarray(pcd_remove_outlier.points)

    # estimate scale
    bounding_box_mesh = mesh.get_axis_aligned_bounding_box()
    diagonal_length_mesh = np.linalg.norm(bounding_box_mesh.get_max_bound() - bounding_box_mesh.get_min_bound())
    diagonal_length_pts = np.linalg.norm(pts_removed.max(axis=0) - pts_removed.min(axis=0))
    scale = diagonal_length_mesh/diagonal_length_pts
    print(f'Estimate initial scale {scale}')

    print("this case does not have ground truth")
    T_co_gt = np.eye(4)

    # generate bunch of initial guess and rank them
    translation_guess = pts_removed.mean(axis=0)
    T_co_init_list = sample_T_oc_init_list(20,translation_guess)
    T_co_init_ranked, _ = rank_T_init(sdf_network,pts_removed,T_co_init_list)

    return pts_removed, mesh, sdf_network, T_co_gt,T_co_init_ranked,scale


def prepare_optimization(case_name):
    if case_name =='owl':
        #rendered dataset
        return owl()
    elif case_name == 'cup_hz':
        # real dataset from hz
        return cup_hz()
    elif case_name[:11] == 'object_rgba': # check the prefix
        # real dataset from ar
        mask_id = case_name[-1]
        return load_kinect_dataset(case_name,mask_id)
    else:
        raise NotImplementedError
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='6DoF Optimizer',
                    description='Optimize the pose of the object based on the sdf network',
                    epilog='type object_name, for example: owl, cup_hz, object_rgba_1')
    parser.add_argument('object_name', type=str, help='name of the object for optimization')   
    args = parser.parse_args()

    case_name = args.object_name
    pts, mesh, sdf_network, T_co_gt,T_co_init,scale = prepare_optimization(case_name)

    config_path = './confs/optimizer.json' # into args
    configs = get_configs(config_path)
    optimizer = Optimizer(sdf_network,configs)



    # feed the ranked initial guess to the optimizer
    T_co_optimized = optimizer.estimate_pose_cam_obj(T_co_init,scale=1/scale,pts=pts)  # scale should be inverse with T_co
    T_co_optimized = T_co_optimized.detach().numpy()


    # get the inverse of the transformation
    T_oc_gt = np.linalg.inv(T_co_gt)
    T_oc_init = np.linalg.inv(T_co_init)
    T_oc_optimized = np.linalg.inv(T_co_optimized)

    # transfer to obj frame        
    pts_in_obj_gt = pts@ T_oc_gt[:3,:3].T+ T_oc_gt[:3,3]
    pts_in_obj_gt_scaled = pts_in_obj_gt*scale
    pts_in_obj_init = pts@ T_oc_init[:3,:3].T + T_oc_init[:3,3]
    pts_in_obj_init_scaled = pts_in_obj_init*scale
    pts_in_obj_optimized = pts@ T_oc_optimized[:3,:3].T + T_oc_optimized[:3,3]
    pts_in_obj_optimized_scaled = pts_in_obj_optimized*scale 

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    pcd_gt = visualize_points(pts_in_obj_gt_scaled)
    pcd_init = visualize_points(pts_in_obj_init_scaled)
    pcd_optimized = visualize_points(pts_in_obj_optimized_scaled)

    pcd_gt.paint_uniform_color([0, 1, 0]) # in green
    pcd_optimized.paint_uniform_color([0, 0, 1]) # in blue
    pcd_init.paint_uniform_color([1, 0, 0]) # in red

    # o3d.visualization.draw_geometries([mesh,pcd_init])
    o3d.visualization.draw_geometries([mesh,pcd_init, pcd_optimized,axis])
    # o3d.visualization.draw([pcd_gt,pcd_optimized,pcd_init, mesh,axis])

    # o3d.visualization.capture_screen_image('./screen.jpg')
    # record the results
    optimization_results = {'T_oc_optimized': T_oc_optimized.tolist(),
                            'T_oc_init': T_oc_init.tolist(),
                            'T_oc_gt': T_oc_gt.tolist()
                            }

    print("optimization done")
    print(f'peak memory used: {torch.cuda.max_memory_allocated()*1e-9} GB')

    optimization_path = './optimization_results.json'
    save_dict(optimization_path,optimization_results)
    print(f'saved results to {optimization_path}')

