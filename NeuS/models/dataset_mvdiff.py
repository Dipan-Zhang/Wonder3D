import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import PIL.Image
from glob import glob
import pdb
import matplotlib.pyplot as plt
from PIL import Image

from models.features.clip_extract import CLIPArgs, extract_clip_features
from models.features.dino_extract import DINOArgs, extract_dino_features
from models.features.pca_colormap import apply_pca_colormap


def camNormal2worldNormal(rot_c2w, camNormal):
    H,W,_ = camNormal.shape
    normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def worldNormal2camNormal(rot_w2c, worldNormal):
    H,W,_ = worldNormal.shape
    normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])

    return normal_img

def trans_normal(normal, RT_w2c, RT_w2c_target):

    normal_world = camNormal2worldNormal(np.linalg.inv(RT_w2c[:3,:3]), normal)
    normal_target_cam = worldNormal2camNormal(RT_w2c_target[:3,:3], normal_world)

    return normal_target_cam

def img2normal(img):
    return (img/255.)*2-1

def normal2img(normal):
    return np.uint8((normal*0.5+0.5)*255)

def norm_normalize(normal, dim=-1):

    normal = normal/(np.linalg.norm(normal, axis=dim, keepdims=True)+1e-6)

    return normal

def RT_opengl2opencv(RT):
     # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)

    return RT

def normal_opengl2opencv(normal):
    H,W,C = np.shape(normal)
    # normal_img = np.reshape(normal, (H*W,C))
    R_bcam2cv = np.array([1, -1, -1], np.float32)
    normal_cv = normal * R_bcam2cv[None, None, :]

    print(np.shape(normal_cv))

    return normal_cv

def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]

def load_a_prediction(root_dir, test_object, imSize, view_types, load_color=False, cam_pose_dir=None, normal_system='front'):

    all_images = []
    all_normals = []
    all_normals_world = []
    all_masks = []
    all_poses = []
    all_w2cs = []
    print(f'camera pose directory {cam_pose_dir}')
    RT_front = np.loadtxt(glob(os.path.join(cam_pose_dir, '*_%s_RT.txt'%( 'front')))[0])   # world2cam matrix
    RT_front_cv = RT_opengl2opencv(RT_front)   # convert normal from opengl to opencv

    for idx, view in enumerate(view_types):
        # originally use masked normal and unmasked rgb
        normal_filepath = os.path.join(root_dir,test_object, 'masked_normals','normals_000_%s.png'%(view)) # here use unmasked
        # Load key frame
        if load_color:  # whether use bg?
            rgb_filepath = os.path.join(root_dir,test_object, 'rgb','rgb_000_%s.png'%(view)) # here use unmasked
            image =np.array(PIL.Image.open(rgb_filepath).resize(imSize))[:, :, ::-1]


        normal = np.array(PIL.Image.open(normal_filepath).resize(imSize))
        mask = normal[:, :, 3] # why mask is relevant with normal???
        normal = normal[:, :, :3]

        RT = np.loadtxt(os.path.join(cam_pose_dir, '000_%s_RT.txt'%(view)))  # world2cam matrix

        normal = img2normal(normal)

        normal[mask==0] = [0,0,0]
        mask = mask> (0.5*255)
        if load_color:
            all_images.append(image)
        
        all_masks.append(mask)
        RT_cv = RT_opengl2opencv(RT)   # convert normal from opengl to opencv
        all_poses.append(inv_RT(RT_cv))   # cam2world
        all_w2cs.append(RT_cv)

        # whether to 
        normal_cam_cv = normal_opengl2opencv(normal)

        if normal_system == 'front':
            normal_world = camNormal2worldNormal(inv_RT(RT_front_cv)[:3, :3], normal_cam_cv)
        elif normal_system == 'self':
            normal_world = camNormal2worldNormal(inv_RT(RT_cv)[:3, :3], normal_cam_cv)
        all_normals.append(normal_cam_cv)
        all_normals_world.append(normal_world)
        
        if not load_color:
            all_images = [normal2img(x) for x in all_normals_world]

    return np.stack(all_images), np.stack(all_masks), np.stack(all_normals), np.stack(all_normals_world), np.stack(all_poses), np.stack(all_w2cs)

def expand_features(features,img_size):
    features = features.permute(0, 3, 1, 2)  # New shape will be [6, 384, 55, 55]

    # Here, mode='bilinear' is used for 2D interpolation; you can also use other modes like 'nearest', 'bicubic', etc.
    expanded_features = F.interpolate(features, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)

    # Step 3: Permute back to [batch_size, height, width, channels]
    expanded_features = expanded_features.permute(0, 2, 3, 1) 

    return expanded_features

def extract_features(root_dir, test_object,feat_type, view_types, normal_system='front', visualize=False):

    """Extract features with support for caching.
    return features
    """
    feat_type_to_extract_fn = {
    "CLIP": extract_clip_features,
    "DINO": extract_dino_features,
    }

    if feat_type == "CLIP":
        raise NotImplementedError
    
    extract_fn = feat_type_to_extract_fn[feat_type]
    image_fnames = []
    for idx, view in enumerate(view_types):
        rgb_filepath = os.path.join(root_dir, test_object, 'rgb','rgb_000_%s.png'%(view))
        image_fnames.append(rgb_filepath)
   
    cache_dir = os.path.join(root_dir,'cache')
    cache_path = cache_dir + f"/{test_object}_features.pt"

    enable_cache= True
    if enable_cache and os.path.exists(cache_path):
        cache_dict = torch.load(cache_path)
        if cache_dict.get("image_fnames") != image_fnames:
            print("Image filenames have changed, cache invalidated... Reextract features from images")
            features = extract_fn(image_fnames, device='cuda')
            return features
        else:
            print(f'load cached feature from {cache_path}')
            return cache_dict["features"]
    else:
        # Cache is invalid or doesn't exist, so extract features
        print(f"Extracting {feat_type} features for {len(image_fnames)} images...")
        features = extract_fn(image_fnames, device='cuda')
        
        if visualize == True:
            feature_pca = []
            for feature in features:
                feature_pca.append(apply_pca_colormap(feature).cpu().numpy())
            plt.figure()
            plt.suptitle("Visualize extracted PCA features")

            for i, (image_path, dino_pca_) in enumerate(zip(image_fnames, feature_pca)):
                plt.subplot(2, len(image_fnames), i + 1)
                plt.imshow(Image.open(image_path))
                plt.title(os.path.basename(image_path))
                plt.axis("off")

                plt.subplot(2, len(image_fnames), len(image_fnames) + i + 1)
                plt.imshow(dino_pca_)
                plt.axis("off")

            plt.tight_layout()
            plt.savefig("demo_extract_features.png")
            print("Saved plot to demo_extract_features.png")
            plt.show()
        

        if enable_cache: # save to cache
            cache_dict = {"image_fnames": image_fnames, "features": features}
            os.makedirs(cache_dir,exist_ok=True)
            torch.save(cache_dict, cache_path)
            print(f"Saved {feat_type} features to cache at {cache_path}")

        return features


    

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.object_name = conf.get_string('object_name')
        self.object_viewidx = conf.get_int('object_viewidx')
        self.imSize = conf['imSize']
        self.load_color = conf['load_color']
        self.stage = conf['stage']
        self.mtype = conf['mtype']
        self.num_views = conf['num_views']

        self.normal_system = conf['normal_system']
        
        self.cam_pose_dir = "./models/fixed_poses/"

        if self.num_views == 4:
            view_types = ['front', 'right', 'back', 'left']
        elif self.num_views == 5:
            view_types = ['front', 'front_right', 'right', 'back', 'left']
        elif self.num_views == 6:
            view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']

        # load data here 
        self.images_np, self.masks_np, self.normals_cam_np, \
        self.normals_world_np ,self.pose_all_np, self.w2c_all_np = load_a_prediction(
                self.data_dir, self.object_name, self.imSize, view_types, self.load_color,
                  self.cam_pose_dir, normal_system=self.normal_system)
        
        # load features
        features = extract_features(self.data_dir,self.object_name,'DINO', view_types, normal_system=self.normal_system) # # [n_images, 55, 55, 384]
        self.features = expand_features(features, self.imSize) # [n_images, H, W, 384] #! get rid of this when oom
        # self.features_rgb_np = apply_pca_colormap(self.features).cpu().numpy() * 255

        self.n_images = self.images_np.shape[0]

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu() / 255.  # [n_images, H, W, 4]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 1]
        self.normals_cam  = torch.from_numpy(self.normals_cam_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.normals_world  = torch.from_numpy(self.normals_world_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.pose_all  = torch.from_numpy(self.pose_all_np.astype(np.float32)).cpu()   # [n_images,3, 4] cam2world

        # self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        self.intrinsic = torch.from_numpy(np.array([
            [self.W/2.0,  0,          self.W / 2.0, 0],
            [0,           self.H/2.0, self.H/ 2.0,  0],
            [0,           0,          1,            0],
            [0,           0,          0,            1]
        ]).astype(np.float32))

        self.intrinsics_all = torch.stack([self.intrinsic]*self.num_views, dim=0).cpu()
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all).cpu()  # [n_images, 4, 4]

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])

        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]

        self.near = 0.2
        self.far = 2.4

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.all_rays = self.prepare_all_rays()
        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        q = torch.stack([(pixels_x/self.W-0.5)*2, (pixels_y/self.H-0.5)*2, torch.zeros_like(pixels_y)], dim=-1) # W, H, 3

        v = torch.stack([torch.zeros_like(pixels_y), torch.zeros_like(pixels_y), torch.ones_like(pixels_y)], dim=-1) # W, H, 3

        # orthogonal projection
        rays_v = v / torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3].cuda(), rays_v[:, :, :, None].cuda()).squeeze()  # W, H, 3
        
        rays_o = torch.matmul(self.pose_all[img_idx, None, None, :3, :3].cuda(), q[:, :, :, None].cuda()).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape).cuda() + rays_o  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu() 
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu() 
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        normal = self.normals_world[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        
        q = torch.stack([(pixels_x / self.W-0.5)*2, (pixels_y / self.H-0.5)*2, torch.zeros_like(pixels_y)], dim=-1).float()  # batch_size, 3
        v = torch.stack([torch.zeros_like(pixels_y), torch.zeros_like(pixels_y), torch.ones_like(pixels_y)], dim=-1).float()

        # q = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float() # bsz, 3
        # q = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], q[:, :, None]).squeeze()  # bsz, 3
        # q[:, 2] = 0
        
        rays_v = v / torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        
        rays_o = torch.matmul(self.pose_all[img_idx, None, :3, :3], q[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) + rays_o # batch_size, 3
        
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, None], normal], dim=-1).cuda()    # batch_size, 10

    def prepare_rays_a_view(self, img_idx):
        """
        Generate random rays at world space from one camera from single image
        """
        # create a 2d pixel grid 
        tx = torch.linspace(0, self.W - 1, self.W)
        ty = torch.linspace(0, self.H - 1, self.H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.reshape(-1).long()
        pixels_y = pixels_y.reshape(-1).long()

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3 -> 4 
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        features = self.features[img_idx][((pixels_y, pixels_x))].to('cpu') # batch_size 1
        normal = self.normals_world[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        
        q = torch.stack([(pixels_x / self.W-0.5)*2, (pixels_y / self.H-0.5)*2, torch.zeros_like(pixels_y)], dim=-1).float()  # batch_size, 3
        v = torch.stack([torch.zeros_like(pixels_y), torch.zeros_like(pixels_y), torch.ones_like(pixels_y)], dim=-1).float() # 001 z-axis
        
        rays_v = v / torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        
        rays_o = torch.matmul(self.pose_all[img_idx, None, :3, :3], q[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) + rays_o # batch_size, 3

        # the similarity between 
        cosines = self.cos(rays_v, normal)

        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, None], normal, cosines[:, None],features], dim=-1)   # batch_size, 10


    def prepare_all_rays(self,):
        all_rays = []
        for idx in range(self.n_images):
            rays = self.prepare_rays_a_view(idx) # from each image
            all_rays.append(rays)
        all_rays = torch.concat(all_rays, dim=0)
        return all_rays

    def __getitem__(self, idx):
        return self.all_rays[idx]
    
    def __len__(self):
        return self.all_rays.shape[0]

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far
    
    def get_near_far(self,):
        return self.near, self.far

    def image_at(self, idx, resolution_level):
        img = self.images_np[idx][:,:,:3]
        # breakpoint()
        return (cv2.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def normal_cam_at(self, idx, resolution_level):
        normal_cam = self.normals_cam_np[idx]
        img = normal2img(normal_cam)
        return (cv2.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def mask_at(self, idx, resolution_level):
        mask = np.uint8(self.masks_np[idx]*255)[:, :, None]
        mask = np.concatenate([mask]*3, axis=-1)
        return (cv2.resize(mask, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
    
    # def feature_at(self,idx,resolution_level):
    #     feature = self.features_rgb_np[idx]
    #     # breakpoint()
    #     return (cv2.resize(feature, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)



# if __name__=='__main__':
#     features = extract_features('/home/stud/zanr/code/tmp/Wonder3D/outputs/cropsize-192-cfg1.0/', 'owl','DINO',256, ['front', 'front_right', 'right', 'back', 'left', 'front_left'], load_color=False, cam_pose_dir=None, normal_system='front')
    # print(features.shape) # ([6, 55, 55, 384]) n_images, h, w, img_emb_dim
    
