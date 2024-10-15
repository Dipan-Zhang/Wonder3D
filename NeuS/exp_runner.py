import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
import wandb
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset_mvdiff import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, FeatureField
from models.renderer import NeuSRenderer, FeatureRender, select_vertices_and_update_triangles
from models.features.pca_colormap import apply_pca_colormap
import pdb
import math


"""
parse input from cli
construct runner class
runner.train()
"""

def ranking_loss(error, penalize_ratio=0.7, type='mean'):
    """
    get rid of some very big error values -> stable convergence
    penalize_ratio close to 1: focusing on the majority of errors
    """
    error, indices = torch.sort(error) # sort increasing
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[: int(penalize_ratio * indices.shape[0])]) # select only ratio part of the errors, smaller errors
    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, data_dir=None):
        """"
        mode: train, save_maps, validate_mesh, interpolate(create novel view)
        is_contine: whether continue unfinished training
        """
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset']['data_dir'] = data_dir
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.conf['train']['batch_size'],
            shuffle=True,
            num_workers=20, # 64 -> 20 according to warning message
        )
        self.iter_step = 1
        self.case_name = case

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq') 
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_feature = self.conf.get_float('train.learning_rate_feature')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        self.sequential_train = self.conf.get_bool('train.sequential_train')
        self.geo_only_iter = self.conf.get_int('train.geo_only_iter')
        self.wandb_mode = self.conf.get_string('train.wandb_mode')

        # Weights
        self.color_weight = self.conf.get_float('train.color_weight')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.normal_weight = self.conf.get_float('train.normal_weight')
        self.sparse_weight = self.conf.get_float('train.sparse_weight')
        self.feature_weight = self.conf.get_float('train.feature_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        # self.feature_network = FeatureNetwork(**self.conf['model.feature_network']).to(self.device) # feature network from LERF
        self.feature_network = FeatureField(**self.conf['model.feature_field']).to(self.device) # feature network borrowed from F3RM
        self.feature_render = FeatureRender()
        params_to_train_slow = []
        # params_to_train_slow += list(self.nerf_outside.parameters())#? why this line got commented?
        params_to_train_slow += list(self.sdf_network.parameters())
        params_to_train_slow += list(self.deviation_network.parameters())
 
        # Set up optimizer: will be updated when sequential training!
        self.optimizer = torch.optim.Adam(
            [{'params': params_to_train_slow, 'lr': self.learning_rate}, {'params': self.color_network.parameters(), 'lr': self.learning_rate * 2}]
        )
        self.renderer = NeuSRenderer(
            self.nerf_outside, self.sdf_network, self.deviation_network, self.color_network, self.feature_network, self.feature_render,**self.conf['model.neus_renderer']
        )

        # Load checkpoint
        latest_model_name = None
        if is_continue: # cont
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_ckpt = True
            self.load_checkpoint(latest_model_name)
        else:
            self.load_ckpt = False

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = wandb.init(project = self.conf.get_string('train.wandb_project_name'),
                                 mode=self.wandb_mode,
                                 config = {
                                    "learning_rate": self.learning_rate,
                                    "learning_rate_feature": self.learning_rate_feature,
                                    "feature_weight": self.feature_weight,
                                 }
                                )
        
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        num_train_epochs = math.ceil(res_step / len(self.dataloader))

        if not self.load_ckpt:
            for epoch in range(num_train_epochs):
                # for iter_i in tqdm(range(res_step)):
                print("epoch ", epoch)
                for iter_i, data in enumerate(self.dataloader):
                    data = data.cuda()

                    rays_o, rays_d, true_rgb, mask, true_normal, cosines, feature_gt = (
                        data[:, :3],
                        data[:, 3:6],
                        data[:, 6:9],
                        data[:, 9:10],
                        data[:, 10:13],
                        data[:, 13:14],
                        data[:, 14:],
                    )
                    near, far = self.dataset.get_near_far()

                    background_rgb = None
                    if self.use_white_bkgd:
                        background_rgb = torch.ones([1, 3])

                    if self.mask_weight > 0.0:
                        mask = (mask > 0.5).float()
                    else:
                        mask = torch.ones_like(mask)

                    cosines[cosines > -0.1] = 0
                    mask = ((mask > 0) & (cosines < -0.1)).to(torch.float32)

                    mask_sum = mask.sum() + 1e-5
                    render_out = self.renderer.render(
                        rays_o, rays_d, near, far, background_rgb=background_rgb, cos_anneal_ratio=self.get_cos_anneal_ratio()
                    )

                    color_fine = render_out['color_fine']
                    s_val = render_out['s_val']
                    cdf_fine = render_out['cdf_fine']
                    gradient_error = render_out['gradient_error']
                    weight_max = render_out['weight_max']
                    weight_sum = render_out['weight_sum']
                    feature = render_out['feature']

                    # Loss
                    color_errors = (color_fine - true_rgb).abs().sum(dim=1) 
                    color_fine_loss = ranking_loss(color_errors[mask[:, 0] > 0])
                    psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
                    eikonal_loss = gradient_error
                    mask_errors = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask, reduction='none')
                    mask_loss = ranking_loss(mask_errors[:, 0], penalize_ratio=0.8)

                    def feasible(key):
                        return (key in render_out) and (render_out[key] is not None)

                    # calculate normal loss
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere'):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1)

                    normal_errors = 1 - F.cosine_similarity(normals, true_normal, dim=1)
                    normal_errors = normal_errors * torch.exp(cosines.abs()[:, 0]) / torch.exp(cosines.abs()).sum()
                    normal_loss = ranking_loss(normal_errors[mask[:, 0] > 0], penalize_ratio=0.9, type='sum')

                    sparse_loss = render_out['sparse_loss']

                    # calculate feature loss
                    feature_loss = F.mse_loss(feature, feature_gt) # feature gt: bs x chanel, 512 x 384,
                    
                    # feature_errors = F.mse_loss(feature, feature_gt,reduction='none').sum(dim=1) # feature gt: bs x chanel, 512 x 384,
                    # feature_loss = ranking_loss(feature_errors[mask[:, 0] > 0])

                    #TODO make this part compatible with train together
                    if self.iter_step == self.geo_only_iter: # switch to feature training
                        for param_group in self.optimizer.param_groups[:2]:  # Freezing the first two groups
                            for param in param_group['params']:
                                param.requires_grad = False
                        self.optimizer = torch.optim.Adam(
                            [{'params': self.feature_network.parameters(), 'lr': self.learning_rate_feature}]
                        )

                    loss = (
                        color_fine_loss * self.color_weight
                        + eikonal_loss * self.igr_weight
                        + sparse_loss * self.sparse_weight
                        + mask_loss * self.mask_weight
                        + normal_loss * self.normal_weight
                        + feature_loss * self.feature_weight
                    )

                    metrics = {
                            'iterstep': self.iter_step,
                            'Loss/loss': loss,
                            'Loss/color_loss': color_fine_loss,
                            'Loss/normal_loss': normal_loss,
                            'Loss/mask_loss': mask_loss,
                            'Loss/eikonal_loss': eikonal_loss,
                            'Loss/feature_loss': feature_loss,
                            'Statistics/s_val': s_val.mean(), 
                            'Statistics/cdf': (cdf_fine[:, :1] * mask).sum() / mask_sum,
                            'Statistics/weight_max': (weight_max * mask).sum() / mask_sum,
                            'Statistics/psnr': psnr,
                            'Statistics/feature_range': feature.max() - feature.min(),  # Debug
                    }
                    self.writer.log(metrics)

                    # # use sequential training: needs 2 optimizers
                    # self.optimizer_geometry.zero_grad()
                    # self.optimizer_feature.zero_grad()

                    # loss.backward()
                    # self.optimizer_geometry.step()
                    # self.optimizer_feature.step()

                    self.optimizer.zero_grad()                   
                    loss.backward()
                    self.optimizer.step()

                    if self.iter_step % self.report_freq == 0:
                        print(
                            'iter:{:8>d} loss = {:4>f} color_ls = {:4>f} eik_ls = {:4>f} normal_ls = {:4>f} mask_ls = {:4>f} feature_ls = {:4>f} sparse_ls = {:4>f} lr={:6>f}'.format(
                                self.iter_step,
                                loss,
                                color_fine_loss,
                                eikonal_loss,
                                normal_loss,
                                mask_loss,
                                feature_loss,
                                sparse_loss,
                                # self.optimizer_geometry.param_groups[0]['lr'],
                                # self.optimizer_feature.param_groups[0]['lr']
                                self.optimizer.param_groups[0]['lr'],
                                # self.optimizer.param_groups[2]['lr']
                            )
                        )
                        print('iter:{:8>d} s_val = {:4>f}'.format(self.iter_step, s_val.mean()))

                    if self.iter_step % self.val_mesh_freq == 0:
                        self.validate_mesh(resolution=256)

                    self.update_learning_rate()

                    self.iter_step += 1

                    if self.iter_step % self.val_freq == 0:
                        self.validate_image(idx=0)
                        self.validate_image(idx=1)
                        self.validate_image(idx=2)
                        self.validate_image(idx=3)

                    if self.iter_step % self.save_freq == 0:
                        self.save_checkpoint()

                    if self.iter_step % len(image_perm) == 0:
                        image_perm = self.get_image_perm()
        else:
            logging.info(f'loaded pretrained ckpt from lastest model and skip geometry training part')

        wandb.finish()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        # for g in self.optimizer_geometry.param_groups:
        #     g['lr'] = self.learning_rate * learning_factor
        
        # for g in self.optimizer_feature.param_groups:
        #     g['lr'] = self.learning_rate_feature * learning_factor

        # Update learning rates for each parameter group
        if self.iter_step < self.geo_only_iter:
            for i, g in enumerate(self.optimizer.param_groups):
                if i == 0:  # params_to_train_slow
                    g['lr'] = self.learning_rate * learning_factor
                elif i == 1:  # color_network parameters
                    g['lr'] = self.learning_rate * 2 * learning_factor
        else:  
            for i, g in enumerate(self.optimizer.param_groups):
                if i == 0:
                    g['lr'] = self.learning_rate_feature * learning_factor


    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.feature_network.load_state_dict(checkpoint['feature_network'])
        self.optimizer_geometry.load_state_dict(checkpoint['optimizer-geometry'])
        self.optimizer_feature.load_state_dict(checkpoint['optimizer-feature'])

        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'feature_network': self.feature_network.state_dict(),
            # 'optimizer-geometry': self.optimizer_geometry.state_dict(),
            # 'optimizer-feature': self.optimizer_feature.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape # 256 x 256
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size) 
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_mask = []
        out_feature = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            # near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('weight_sum'):
                out_mask.append(render_out['weight_sum'].detach().clip(0, 1).cpu().numpy())

            if feasible('feature'):
                out_feature.append(render_out['feature'].detach()) # 512 x 384
            del render_out

        # out_rgb_fine 128 x 512 x 3
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)  # concatenate  (65536, 3) -> reshape  (256, 256, 3, 1)

        mask_map = None
        if len(out_mask) > 0:
            mask_map = (np.concatenate(out_mask, axis=0).reshape([H, W, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255) # (256, 256, 3, 1)

        feature_img = None # feature image after PCA together with gt feature
        if len(out_feature) > 0:
            temp_img = (torch.cat(out_feature, axis=0).reshape([H, W, 384, -1])) # [h, w, c, n] [256, 256, 384, 1] 
            temp_img_re = temp_img.permute(3,0,1,2) # n x h x w x c(384)
            imgs = torch.cat((temp_img_re,self.dataset.features[idx].unsqueeze(0)),0) # [n, h, w, c(3)]
            feature_imgs_pca = apply_pca_colormap(imgs) # [n, h, w, c(3)]
            
            feature_img_pca = feature_imgs_pca[0, :, :, :].detach().cpu().numpy() # [h, w, c]
            gt_feature_img_pca = feature_imgs_pca[1, :, :, :].detach().cpu().numpy() # [h, w, c]

            feature_img = np.concatenate([feature_img_pca, gt_feature_img_pca])[:, :, ::-1] # [h, w, c]
            feature_img = (feature_img * 256).clip(0, 255).astype(np.uint8) 

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'feature'), exist_ok=True)

        log_images = []
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                    np.concatenate(
                        [
                            img_fine[..., i], # [h, w, c]
                            self.dataset.image_at(idx, resolution_level=resolution_level),
                            self.dataset.mask_at(idx, resolution_level=resolution_level),
                        ]
                    ),
                )
                log_images.append(wandb.Image(img_fine[..., i][...,[2,1,0]], caption=f"rgb"))

            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                    np.concatenate([normal_img[..., i], self.dataset.normal_cam_at(idx, resolution_level=resolution_level)])[:, :, ::-1],
                )
                log_images.append(wandb.Image(normal_img[..., i], caption=f"normal image"))

            if len(out_mask) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}_mask.png'.format(self.iter_step, i, idx)), mask_map[..., i])
                log_images.append(wandb.Image(mask_map[..., i], caption=f"mask image"))
            
            if len(out_feature) > 0:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'feature', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                    feature_img,
                )
                log_images.append(wandb.Image(feature_img, caption=f"feature image"))
            wandb.log({f"viewpoint_{idx}": log_images})
    

    #? functionality ??? 
    def save_maps(self, idx, img_idx, resolution_level=1):
        view_types = ['front', 'back', 'left', 'right']
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_mask = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            # near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('weight_sum'):
                out_mask.append(render_out['weight_sum'].detach().clip(0, 1).cpu().numpy())

            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)

        mask_map = None
        if len(out_mask) > 0:
            mask_map = (np.concatenate(out_mask, axis=0).reshape([H, W, 1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            world_normal_img = (normal_img.reshape([H, W, 3]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'coarse_maps'), exist_ok=True)
        img_rgba = np.concatenate([img_fine[:, :, ::-1], mask_map], axis=-1)
        normal_rgba = np.concatenate([world_normal_img[:, :, ::-1], mask_map], axis=-1)

        cv.imwrite(os.path.join(self.base_exp_dir, 'coarse_maps', "normals_mlp_%03d_%s.png" % (img_idx, view_types[idx])), img_rgba)
        cv.imwrite(os.path.join(self.base_exp_dir, 'coarse_maps', "normals_grad_%03d_%s.png" % (img_idx, view_types[idx])), normal_rgba)

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        breakpoint()
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            # near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        # TODO add feature image

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        # novel_images={
        #     'rgb':img_fine,
        #     'feature': img_feature
        # }
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles, vertex_colors = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        # selected_vertices, selected_triangles = select_vertices_and_update_triangles(vertices, triangles) # TODO can use this to reduce memory usage?
        feature_vertices_rgb = self.renderer.fuse_feature2mesh(vertices) 
        feature_vertices_rgb_np = (feature_vertices_rgb.cpu().detach().numpy() * 255).clip(0, 255)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        # export as obj
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', f'{self.case_name}_{self.iter_step}.obj'))

        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=feature_vertices_rgb_np)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', f'{self.case_name}_feature_{self.iter_step}.obj'))
        del mesh
        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0, img_idx_1, np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5, resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir, '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)), fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.data_dir)

    if args.mode == 'train':
        runner.train()
        runner.validate_mesh(world_space=False, resolution=256, threshold=args.mcube_threshold)
    elif args.mode == 'save_maps':
        for i in range(4): 
            runner.save_maps(idx=i, img_idx=runner.dataset.object_viewidx)
    elif args.mode == 'validate_mesh':
        # TODO load trained checkpoint?
        runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
