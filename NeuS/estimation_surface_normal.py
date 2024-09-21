import numpy as np
import cv2
# from diffuser_utils.dataset_utils import *
import open3d as o3d
import torch
import os 
import argparse

def fit_plane_from_points(points, threshold=0.01, ransac_n=3, num_iterations=2000):
    # Segments a plane in the point cloud using the RANSAC algorithm. return a tuple (4x1 vect, index)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = o3d.geometry.PointCloud.segment_plane(
        pcd,
        distance_threshold=threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return plane_model

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

def visualize_transformation(T):
    axis_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    axis_o3d.transform(T)
    return axis_o3d


# !!! here really important
def RT_opengl2opencv(RT):
     # Build the coordinate transform matrix from world to computer vision camera
    new_RT = np.eye(4)
    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t
    new_RT[:3, :3] = R_world2cv
    new_RT[:3, 3] = t_world2cv

    return new_RT


def Rx(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((1, 0, 0, 0), (0, c, -s, 0), (0, s, c, 0), (0, 0, 0, 1)))
    return R


def split_underscore(input_str):
    # split the input string into the last part and the rest by underscore
    end = input_str.split('_')[-1]
    rest = '_'.join(input_str.split('_')[:-1])
    return rest, end


def main():
    parser = argparse.ArgumentParser(
                    prog='Surface normal Estimator',
                    description='Estimate the surface normal for the table detected in the dataset to generated fixed camera poses',
                    epilog='type object_name, for example: table_1_far_1 object_rgba_1')
    parser.add_argument('object_name', type=str, help='name of the object for optimization')   
    parser.add_argument('-v','--visualize', action='store_true', default=False, help='visualize the camera poses during computation')
    args = parser.parse_args()
    object_name = args.object_name
    visualize_flag = args.visualize
 
    dataset_name, mask_id = split_underscore(object_name)

    dataset_path = '../customized_data'
    depth_path = f"{dataset_path}/{dataset_name}/depth/00000.png"
    color_path = f"{dataset_path}/{dataset_name}/color/00000.jpg"
    desk_mask_path = f"{dataset_path}/{dataset_name}/desk_mask.png"
    obj_mask_path = f"{dataset_path}/{dataset_name}/masks/mask_{mask_id}.png"

    color = cv2.imread(color_path)[..., ::-1] / 255
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
    desk_mask = cv2.imread(desk_mask_path, cv2.IMREAD_UNCHANGED)
    obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_UNCHANGED)
    obj_depth_center = np.median(depth[obj_mask > 0])
    print(obj_depth_center)
    valid_mask = (desk_mask > 0) | (obj_mask > 0)

    intrinsics = np.array(
        [
            [608.7960205078125, 0, 632.1019287109375],
            [0, 608.9515380859375, 365.985595703125],
            [0, 0, 1],
        ]
    )

    points_fit, _ = backproject(depth, intrinsics, desk_mask > 0, False)
    points_obj, _ = backproject(depth, intrinsics, obj_mask > 0, False)
    points, scene_ids = backproject(depth, intrinsics, valid_mask > 0, False)


    ##### Method 1: RANSAC
    plane_model = fit_plane_from_points(points_fit)
    plane_dir = -plane_model[:3] # surface normal

    colors = color[scene_ids[0], scene_ids[1]]
    pcd = visualize_points(points, colors)

    axis_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    T_cam_obj = np.eye(4) # T_cam_obj/ surface 
    T_cam_obj[:3, 2] = plane_dir # new z = plane z 
    T_cam_obj[:3, 1] = -np.cross([1, 0, 0], plane_dir) #new_y = x cross z
    T_cam_obj[:3, 3] = points_obj.mean(axis=0)

    T_obj_cam = np.linalg.inv(T_cam_obj)
    axis_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    axis_obj.transform(T_cam_obj)
    points_in_obj = points @ T_obj_cam[:3, :3].T + T_obj_cam[:3, 3]
    pcd_in_obj = visualize_points(points_in_obj, colors)

    # initial visualization 
    if visualize_flag:
        o3d.visualization.draw([pcd, axis_obj,axis_o3d],'camera and pointcloud') 


    #### Load the pose from neus
    fixed_poses_folder = 'fixed_poses'
    T_c1o = np.eye(4) 
    T_c1o[:3] = np.loadtxt(f"/home/stud/zanr/code/tmp/Wonder3D/instant-nsr-pl/datasets/{fixed_poses_folder}/000_front_RT.txt")

    T_c2o = np.eye(4) 
    T_c2o[:3] = np.loadtxt(f"/home/stud/zanr/code/tmp/Wonder3D/instant-nsr-pl/datasets/{fixed_poses_folder}/000_front_left_RT.txt")

    T_c3o = np.eye(4) 
    T_c3o[:3] = np.loadtxt(f"/home/stud/zanr/code/tmp/Wonder3D/instant-nsr-pl/datasets/{fixed_poses_folder}/000_right_RT.txt")

    T_c4o = np.eye(4) 
    T_c4o[:3] = np.loadtxt(f"/home/stud/zanr/code/tmp/Wonder3D/instant-nsr-pl/datasets/{fixed_poses_folder}/000_back_RT.txt")

    T_c5o = np.eye(4) 
    T_c5o[:3] = np.loadtxt(f"/home/stud/zanr/code/tmp/Wonder3D/instant-nsr-pl/datasets/{fixed_poses_folder}/000_front_right_RT.txt")

    T_c6o = np.eye(4) 
    T_c6o[:3] = np.loadtxt(f"/home/stud/zanr/code/tmp/Wonder3D/instant-nsr-pl/datasets/{fixed_poses_folder}/000_left_RT.txt")

    fixed_poses_fnames = ['000_front_RT','000_front_left_RT','000_right_RT','000_back_RT','000_front_right_RT','000_left_RT']

    # Transform pose from OpenGl to OpenCV
    T_c1o = RT_opengl2opencv(T_c1o)
    T_c2o = RT_opengl2opencv(T_c2o)
    T_c3o = RT_opengl2opencv(T_c3o)
    T_c4o = RT_opengl2opencv(T_c4o)
    T_c5o = RT_opengl2opencv(T_c5o)
    T_c6o = RT_opengl2opencv(T_c6o)

    T_oc1 = np.linalg.inv(T_c1o)
    T_oc2 = np.linalg.inv(T_c2o)
    T_oc3 = np.linalg.inv(T_c3o)
    T_oc4 = np.linalg.inv(T_c4o)
    T_oc5 = np.linalg.inv(T_c5o)
    T_oc6 = np.linalg.inv(T_c6o)

    # # Visualize the axis before changing anything
    axis_c1 = visualize_transformation(T_oc1) # visualize the camera pose in base frame
    axis_c2 = visualize_transformation(T_oc2)
    axis_c3 = visualize_transformation(T_oc3)
    axis_c4 = visualize_transformation(T_oc4)
    axis_c5 = visualize_transformation(T_oc5)
    axis_c6 = visualize_transformation(T_oc6)
    # o3d.visualization.draw([axis_c1, axis_c2, axis_c3, axis_c4, axis_c5, axis_c6, pcd, axis_obj,axis_o3d], 'loaded T from nrl') # camera pose in base frame

    ##### All camera pose rotate around the plane axis
    T_offset = np.eye(4)
    T_offset[:3, :3] = T_cam_obj[:3, :3]
    T_offset[:3, 2] = -T_cam_obj[:3, 2]
    T_offset[:3, 1] = -T_cam_obj[:3, 1]
    T_offset = np.linalg.inv(T_offset)
    # T_offset = np.eye(4)
    T_c1o = T_c1o @ T_offset
    T_c2o = T_c2o @ T_offset
    T_c3o = T_c3o @ T_offset
    T_c4o = T_c4o @ T_offset
    T_c5o = T_c5o @ T_offset
    T_c6o = T_c6o @ T_offset


    # for visualization
    T_oc1_adjusted = np.linalg.inv(T_c1o)   
    T_oc2_adjusted = np.linalg.inv(T_c2o)
    T_oc3_adjusted = np.linalg.inv(T_c3o)
    T_oc4_adjusted = np.linalg.inv(T_c4o)
    T_oc5_adjusted = np.linalg.inv(T_c5o)
    T_oc6_adjusted = np.linalg.inv(T_c6o)


    axis_c1_c = visualize_transformation(T_oc1_adjusted) # visualize the camera pose in base frame
    axis_c2_c = visualize_transformation(T_oc2_adjusted)
    axis_c3_c = visualize_transformation(T_oc3_adjusted)
    axis_c4_c = visualize_transformation(T_oc4_adjusted)
    axis_c5_c = visualize_transformation(T_oc5_adjusted)
    axis_c6_c = visualize_transformation(T_oc6_adjusted)
    if visualize_flag:
        o3d.visualization.draw([axis_c1_c, axis_c2_c, axis_c3_c, axis_c4_c, axis_c5_c, axis_c6_c, pcd, axis_obj,axis_o3d])

    T_co_save_list = [T_c1o,T_c2o,T_c3o,T_c4o,T_c5o,T_c6o]


    cam_pose_save_dir = f"/home/stud/zanr/code/tmp/Wonder3D/instant-nsr-pl/datasets/fixed_poses_{dataset_name}/" # instant-nsr-pl fixed poses
    os.makedirs(cam_pose_save_dir, exist_ok=True)

    ##### save the adjusted poses
    # T_co_list = [T_c1o, T_c2o, T_c3o, T_c4o, T_c5o, T_c6o]
    for idx, name in enumerate(fixed_poses_fnames):
        # T = T_co_tosave_list[idx]
        T_co = T_co_save_list[idx]
        T_co_ = RT_opengl2opencv(T_co)[:3] # back to opengl
        np.savetxt(os.path.join(cam_pose_save_dir, name + ".txt"), T_co_)


    ### visualize the saved poses
    if visualize_flag:
        file_names = os.listdir()
        axises =[]
        for file_name in fixed_poses_fnames:
            T_co = np.eye(4)
            T_co[:3] = np.loadtxt(os.path.join(cam_pose_save_dir, file_name + ".txt"))
            T_co = RT_opengl2opencv(T_co) # to opencv   
            axis = visualize_transformation(np.linalg.inv(T_co)) # visualize T_oc
            axises.append(axis)
        o3d.visualization.draw(axises + [axis_o3d], 'double check saved poses')



if __name__ =='__main__':
    main()