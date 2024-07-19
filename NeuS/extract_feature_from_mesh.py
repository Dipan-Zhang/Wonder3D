import torch
import numpy as np
import open3d as o3d
import torch.functional
from gedi import GeDi
import torch.nn as nn
import torch.functional as F
import math


def sample_point_cloud_from_mesh(mesh_path, number_of_points=100000):
    # Load the mesh from the GLB file
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Check if the mesh has vertex normals, if not, compute them
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Sample points uniformly from the mesh
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    
    return pcd

def read_pointcloud(pcd_path):
    #if pcd_path  exist
        # pcd = o3d.read_pcd
    # else:
    #     pcd = sample_point_cloud_from_mesh(pcd_path)
    pass

def visualize_point_cloud(pcd):
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def extract_geo_feature(pcd=None, config=None):
    if config == None:
        config = {
          'dim': 32,                                            # descriptor output dimension
          'samples_per_batch': 500,                             # batches to process the data on GPU
          'samples_per_patch_lrf': 4000,                        # num. of point to process with LRF
          'samples_per_patch_out': 512,                         # num. of points to sample for pointnet++
          'r_lrf': .5,                                          # LRF radius
          'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar',   # path to checkpoint
        #   'voxel_size': 0.01,
        #   'patches_per_pair':5000,
        }  
    voxel_size = .01
    patches_per_pair = 5000

    # initialising class
    gedi = GeDi(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('You are using the following device: ', device)

    # getting a pair of point clouds
    if pcd==None: # if not passing pcd directly, read from local storage
        pcd0 = o3d.io.read_point_cloud('data/assets/threed_match_7-scenes-redkitchen_cloud_bin_0.ply')
        pcd1 = o3d.io.read_point_cloud('data/assets/threed_match_7-scenes-redkitchen_cloud_bin_5.ply')

    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])

    # estimating normals (only for visualisation)
    pcd0.estimate_normals()
    pcd1.estimate_normals()
    # o3d.visualization.draw_geometries([pcd0, pcd1])

    # randomly sampling some points from the point cloud
    inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False) # 5000
    inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False) # 5000

    pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float() # 5000 x 3 
    pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

    # applying voxelisation to the point cloud
    pcd0 = pcd0.voxel_down_sample(voxel_size)
    pcd1 = pcd1.voxel_down_sample(voxel_size)

    # pcd in tensor
    _pcd0 = torch.tensor(np.asarray(pcd0.points)).float() 
    _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

    # computing descriptors
    pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0) # 5000x32
    pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)

    # sample some points from source pcd, later chose multiple points
    inds_source = np.random.choice(pts0.shape[0], 1, replace=False) # 1/ 5000

    # find the correspondence point nin pcd1 
    breakpoint()
    pcd0_desc_source = np.repeat(pcd0_desc[inds_source], pcd0_desc.shape[0], axis=0)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(pcd0_desc_source, pcd1_desc)
    pcd1_target_ind = torch.argmax(output)

    #target coordiante
    pt_source = torch.tensor(np.asarryu(pcd0.points)[inds_source]).float()
    pt_target =  torch.tensor(np.asarray(pcd1.points)[pcd1_target_ind]).float()

    

    """for ransac"""
    # preparing format for open3d ransac
    # pcd0_dsdv = o3d.pipelines.registration.Feature()
    # pcd1_dsdv = o3d.pipelines.registration.Feature()

    # pcd_dsdv.data = pcd_desc.T
    # pcd1_dsdv.data = pcd1_desc.T

    # _pcd0 = o3d.geometry.PointCloud()
    # _pcd0.points = o3d.utility.Vector3dVector(pts0)
    # _pcd1 = o3d.geometry.PointCloud()
    # _pcd1.points = o3d.utility.Vector3dVector(pts1)

    # # applying ransac
    # est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     _pcd0,
    #     _pcd1,
    #     pcd0_dsdv,
    #     pcd1_dsdv,
    #     mutual_filter=True,
    #     max_correspondence_distance=.02,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #     ransac_n=3,
    #     checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
    #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
    #     criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

    # applying estimated transformation
    # pcd0.transform(est_result01.transformation)
    # o3d.visualization.draw_geometries([pcd0, pcd1])

if __name__ == "__main__":
    mesh_path = "./exp/neus/owl/meshes/owl_mesh.glb"  # Replace with your actual mesh file path
    number_of_points = 10000  # Adjust the number of points as needed

    # Sample dense point cloud from mesh
    # point_cloud = sample_point_cloud_from_mesh(mesh_path, number_of_points)

    # # Visualize the sampled point cloud
    # visualize_point_cloud(point_cloud)

    # save to file
    
    extract_geo_feature()

