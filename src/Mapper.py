import os
import shutil
import traceback
import subprocess
import time
import cv2
import numpy as np
import open3d as o3d
import torch
import math

from ast import literal_eval
from colorama import Fore, Style
from torch.autograd import Variable
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.common import (get_camera_from_tensor, get_samples, get_samples_with_pixel_grad,
                        get_tensor_from_camera, random_select, setup_seed,
                        as_intrinsics_matrix, get_npc_input_pcl,
                        preprocess_point_cloud, execute_global_registration, refine_registration)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.utils.Logger import Logger

from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d
from pytorch_msssim import ms_ssim

import wandb


class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam, coarse_mapper=False
                 ):

        self.cfg = cfg
        self.args = args
        self.coarse_mapper = coarse_mapper
        self.idx = slam.idx
        self.nice = slam.nice
        self.output = slam.output
        self.verbose = slam.verbose
        self.ckptsdir = slam.ckptsdir
        self.renderer = slam.renderer_map
        self.renderer.sigmoid_coefficient = cfg['rendering']['sigmoid_coef_mapper']
        self.npc = slam.npc
        # add a coarse level npc
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.gt_c2w_list = slam.gt_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame
        self.exposure_feat_shared = slam.exposure_feat
        self.exposure_feat = self.exposure_feat_shared[0].clone(
        ).requires_grad_()

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']
        self.pts_along_ray = cfg['pts_along_ray']
        self.wandb = cfg['wandb']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.project_name = cfg['project_name']
        self.project_name_with_gt = cfg['project_name_with_gt']
        self.use_normals = cfg['use_normals']
        self.use_view_direction = cfg['use_view_direction']
        self.use_dynamic_radius = cfg['use_dynamic_radius']

        # self.dynamic_r_add = dict.fromkeys(cfg['pointcloud']['radius_hierarchy'].keys(), None)
        # self.dynamic_r_query = dict.fromkeys(cfg['pointcloud']['radius_hierarchy'].keys(), None)
        self.dynamic_r_add = {}
        self.dynamic_r_query = {}
        self.dynamic_r_add_fine = None
        self.dynamic_r_query_fine = None
        self.dynamic_r_add_mid = None
        self.dynamic_r_query_mid = None

        self.use_bound = cfg['use_bound']
        self.encode_exposure = cfg['model']['encode_exposure']

        self.radius_hierarchy = cfg['pointcloud']['radius_hierarchy']
        
        #self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio']
        #self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio']
        
        self.geo_iter_ratio = cfg['mapping']['geo_iter_ratio']
        self.col_iter_ratio = cfg['mapping']['col_iter_ratio']
        
        self.radius_query_ratio = cfg['pointcloud']['radius_query_ratio']
        self.color_grad_threshold = cfg['pointcloud']['color_grad_threshold']
        self.eval_img = cfg['rendering']['eval_img']

        self.device = cfg['mapping']['device']
        self.fix_geo_decoder_mid = cfg['mapping']['fix_geo_decoder_mid']
        self.fix_geo_decoder_fine = cfg['mapping']['fix_geo_decoder_fine']
        self.fix_color_decoder = cfg['mapping']['fix_color_decoder']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.BA = False
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.pixels_adding = cfg['mapping']['pixels_adding']
        self.pixels_based_on_color_grad = cfg['mapping']['pixels_based_on_color_grad']
        self.pixels_based_on_normal_grad = cfg['mapping']['pixels_based_on_normal_grad']
        self.num_joint_iters = cfg['mapping']['iters']
        self.mid_iter_ratio = cfg['mapping']['mid_iter_ratio']
        self.geo_iter_first = cfg['mapping']['geo_iter_first']
        self.iters_first = cfg['mapping']['iters_first']
        self.clean_mesh = cfg['meshing']['clean_mesh']
        self.every_frame = cfg['mapping']['every_frame']
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.geo_iter_ratio = cfg['mapping']['geo_iter_ratio']
        self.vis_inside = cfg['mapping']['vis_inside']
        self.mesh_coarse_level = cfg['meshing']['mesh_coarse_level']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        self.local_correction = cfg['mapping']['local_correction']
        self.correction_every = cfg['mapping']['correction_every']
        self.end_correction = cfg['mapping']['end_correction']
        self.more_iters_when_adding = cfg['mapping']['more_iters_when_adding']
        self.use_kf_selection = cfg['mapping']['use_kf_selection']
        self.kf_trans_thre = cfg['mapping']['kf_trans_thre']
        self.kf_rot_thre = cfg['mapping']['kf_rot_thre']
        self.frustum_edge = cfg['mapping']['frustum_edge']
        self.filter_before_add_points = cfg['mapping']['filter_before_add_points']
        self.save_ckpts = cfg['mapping']['save_ckpts']
        self.crop_edge = 0 if cfg['cam']['crop_edge'] is None else cfg['cam']['crop_edge']
        self.save_rendered_image = cfg['mapping']['save_rendered_image']
        self.min_iter_ratio = cfg['mapping']['min_iter_ratio']

        

        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        if self.nice:
            if coarse_mapper:
                self.keyframe_selection_method = 'global'  

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.logger = Logger(cfg, args, self)
        self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                     verbose=self.verbose, device=self.device, wandb=self.wandb,
                                     vis_inside=self.vis_inside, total_iters=self.num_joint_iters,
                                     img_dir=os.path.join(self.output, 'rendered_image'))
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        # self.npc_geo_feats = dict.fromkeys(self.radius_hierarchy.keys(), {})
        # self.npc_col_feats = dict.fromkeys(self.radius_hierarchy.keys(), {})
        self.npc_geo_feats_fine = None
        self.npc_geo_feats_mid = None
        self.npc_col_feats = None

    def filter_point_before_add(self, rays_o, rays_d, gt_depth, prev_c2w):
        with torch.no_grad():
            points = rays_o[..., None, :] + \
                rays_d[..., None, :] * gt_depth[..., None, None]
            points = points.reshape(-1, 3).cpu().numpy()
            H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy

            if torch.is_tensor(prev_c2w):
                prev_c2w = prev_c2w.cpu().numpy()
            w2c = np.linalg.inv(prev_c2w)
            ones = np.ones_like(points[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [points, ones], axis=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c@homo_vertices
            cam_cord = cam_cord_homo[:, :3]
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)

            edge = 0
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
        return torch.from_numpy(~mask).to(self.device).reshape(-1)

    def get_mask_from_c2w(self, c2w, depth_np, level):
        """
        Frustum feature selection based on current camera pose and depth image.
        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame. for each (x,y)<->(width,height)

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)

        # concatenate multiple levels?
        
        points = np.array(self.npc.cloud_pos(level=level)).reshape(-1, 3)
        
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        # make the axis consistent and let the frustum feature selection get the correct features.
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                uv[i:i+remap_chunk, 0],
                                uv[i:i+remap_chunk, 1],
                                interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
       
        depths = np.concatenate(depths, axis=0)

        edge = self.frustum_edge  # crop here on width and height
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        points = points[mask]

        return np.where(mask)[0].tolist()


    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=8, pixels=200):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels,
            H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device, depth_filter=True)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)
            cam_cord_homo = w2c@homo_vertices
            cam_cord = cam_cord_homo[:, :3]
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            #cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w,
                     est_normal, keyframe_dict, keyframe_list, cur_c2w, color_refine=False):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            est_normal (tensor): estimated normal map for current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list of keyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 
            prev_c2w (tensor): est_c2w of last mapping frame.

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        npc = self.npc
        cfg = self.cfg
        device = self.device
        init = True if idx == 0 else False
        bottom = torch.tensor([0, 0, 0, 1.0], device=self.device).reshape(1, 4)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                num = self.mapping_window_size-2
                optimize_frame = list(
                    range(max(0, len(self.keyframe_dict)-1-num), len(self.keyframe_dict)-1))
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                else:
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        decoders_para_list = []
        color_pcl_para = []
        geo_pcl_para_fine = []
        geo_pcl_para_mid = []
        gt_depth_np = cur_gt_depth.cpu().numpy()
        gt_depth = cur_gt_depth.to(device)
        gt_color = cur_gt_color.to(device)

        if idx == 0:
            add_pts_num = torch.clamp(self.pixels_adding * ((gt_depth.median()/2.5)**2),
                                      min=self.pixels_adding, max=self.pixels_adding*3).int().item()
        else:
            add_pts_num = self.pixels_adding
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
            0, H, 0, W, add_pts_num,
            H, W, fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device, depth_filter=True, return_index=True)
        has_est_normal = est_normal is not None
        if has_est_normal:
            batch_est_normal = est_normal[j, i, :]
        
        if not color_refine:
            frame_pts_add = 0
            if self.filter_before_add_points:
                if idx != 0:
                    # make sure add enough points to the non-overlapping area

                    # insert here additional radius for hierarchy levels
                    # adding points to all levels of point clouds
                    mask_add = self.filter_point_before_add(
                        batch_rays_o, batch_rays_d, batch_gt_depth, self.prev_c2w)
                    _fine = self.npc.add_neural_points(batch_rays_o[mask_add], batch_rays_d[mask_add],
                                                   batch_gt_depth[mask_add], batch_gt_color[mask_add],
                                                   normals=batch_est_normal[mask_add] if has_est_normal else None,
                                                   dynamic_radius=self.dynamic_r_add['fine'][j, i][mask_add] if self.use_dynamic_radius else None, level='fine',idx=idx)
                    _mid = self.npc.add_neural_points(batch_rays_o[mask_add], batch_rays_d[mask_add],
                                                   batch_gt_depth[mask_add], batch_gt_color[mask_add],
                                                   normals=batch_est_normal[mask_add] if has_est_normal else None,
                                                   dynamic_radius=self.dynamic_r_add['mid'][j, i][mask_add] if self.use_dynamic_radius else None, level='mid',idx=idx)
                    print(f'{_fine} locations to add points in non-overlapping area of fine level.')
                    print(f'{_mid} locations to add points in non-overlapping area of mid level.')
                    frame_pts_add += _fine

                    # try add points to overlapped area too
                    batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
                        0, H, 0, W, int(1000),
                        H, W, fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device, depth_filter=True, return_index=True)
                    batch_est_normal = est_normal[j,
                                                  i, :] if has_est_normal else None
                    mask_add = self.filter_point_before_add(
                        batch_rays_o, batch_rays_d, batch_gt_depth, self.prev_c2w)
                    _fine = self.npc.add_neural_points(batch_rays_o[~mask_add], batch_rays_d[~mask_add],
                                                   batch_gt_depth[~mask_add], batch_gt_color[~mask_add],
                                                   normals=batch_est_normal[~mask_add] if has_est_normal else None,
                                                   dynamic_radius=self.dynamic_r_add['fine'][j, i][~mask_add] if self.use_dynamic_radius else None, level='fine',idx=idx)
                    _mid = self.npc.add_neural_points(batch_rays_o[~mask_add], batch_rays_d[~mask_add],
                                batch_gt_depth[~mask_add], batch_gt_color[~mask_add],
                                normals=batch_est_normal[~mask_add] if has_est_normal else None,
                                dynamic_radius=self.dynamic_r_add['mid'][j, i][~mask_add] if self.use_dynamic_radius else None, level='mid',idx=idx)
                    print(f'{_fine} locations to add points in overlapping area of fine level.')
                    print(f'{_mid} locations to add points in overlapping area of mid level.')
                    frame_pts_add += _fine
                else:
                    
                    _fine = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, normals=batch_est_normal if has_est_normal else None,
                                dynamic_radius=self.dynamic_r_add['fine'][j, i] if self.use_dynamic_radius else None, level='fine',idx=idx)
                    _mid = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, normals=batch_est_normal if has_est_normal else None,
                                dynamic_radius=self.dynamic_r_add['mid'][j, i] if self.use_dynamic_radius else None, level='mid',idx=idx)
                    print(f'{_fine} locations to add points in fine level.')
                    print(f'{_mid} locations to add points in mid level.')
                    frame_pts_add += _fine
            else:
                _fine = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, normals=batch_est_normal if has_est_normal else None,
                                               dynamic_radius=self.dynamic_r_add['fine'][j, i] if self.use_dynamic_radius else None, level='fine',idx=idx)
                _mid = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, normals=batch_est_normal if has_est_normal else None,
                                               dynamic_radius=self.dynamic_r_add['mid'][j, i] if self.use_dynamic_radius else None, level='mid',idx=idx)
                print(f'{_fine} locations to add points in fine level.')
                print(f'{_mid} locations to add points in mid level.')
                frame_pts_add += _fine

            if self.pixels_based_on_color_grad > 0:
                num_tuple = (self.pixels_based_on_color_grad,
                             self.pixels_based_on_normal_grad)
                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples_with_pixel_grad(
                    0, H, 0, W, num_tuple,
                    H, W, fx, fy, cx, cy, cur_c2w, gt_depth, gt_color, self.device, est_normal,
                    depth_filter=True, return_index=True)
                batch_est_normal = est_normal[j,
                                              i, :] if has_est_normal else None
                _fine = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                                               normals=batch_est_normal if has_est_normal else None, is_pts_grad=True
                                               , dynamic_radius=self.dynamic_r_add['fine'][j, i] if self.use_dynamic_radius else None, level='fine',idx=idx)
                _mid = self.npc.add_neural_points(batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                                               normals=batch_est_normal if has_est_normal else None, is_pts_grad=True
                                               , dynamic_radius=self.dynamic_r_add['mid'][j, i] if self.use_dynamic_radius else None, level='mid',idx=idx)
                print(f'{_fine} locations to add points based on gradient of fine level.')
                print(f'{_mid} locations to add points based on gradient of mid level.')
                frame_pts_add += _fine

        # clone all point feature from shared npc, (N_points, c_dim)
        npc_geo_feats_fine = self.npc.get_geo_feats('fine')
        npc_col_feats_fine = self.npc.get_col_feats('fine')
        npc_geo_feats_mid = self.npc.get_geo_feats('mid')
        npc_col_feats_mid = self.npc.get_col_feats('mid')
        
      
        self.cloud_normals_fine = torch.tensor(
                self.npc.cloud_normal(level='fine'), device=self.device)

        self.cloud_normals_mid = torch.tensor(
                self.npc.cloud_normal(level='mid'), device=self.device)
        
        self.cloud_normals = dict.fromkeys(self.radius_hierarchy.keys(), None)
        self.cloud_normals['mid'] = self.cloud_normals_mid
        self.cloud_normals['fine'] = self.cloud_normals_fine
        
        if self.nice:
            if self.frustum_feature_selection:  # required if not color_refine
                masked_c_grad = {}
                mask_c2w = cur_c2w

                # two indices for geo&color feature? geo->mid level, color->fine level
                indices_geo_fine = self.get_mask_from_c2w(mask_c2w, gt_depth_np,level='fine')
                indices_col_fine = self.get_mask_from_c2w(mask_c2w, gt_depth_np,level='fine')
                indices_geo_mid = self.get_mask_from_c2w(mask_c2w, gt_depth_np,level='mid')
                indices_col_mid = self.get_mask_from_c2w(mask_c2w, gt_depth_np,level='mid')
                
                geo_pcl_grad_fine = npc_geo_feats_fine[indices_geo_fine].clone(
                ).detach().requires_grad_(True)

                geo_pcl_grad_mid = npc_geo_feats_mid[indices_geo_mid].clone(
                ).detach().requires_grad_(True)


                masked_c_grad['indices_geo_fine'] = indices_geo_fine
                masked_c_grad['indices_geo_mid'] = indices_geo_mid
                color_pcl_grad_fine = npc_col_feats_fine[indices_col_fine].clone(
                ).detach().requires_grad_(True)
                color_pcl_grad_mid = npc_col_feats_mid[indices_col_mid].clone(
                ).detach().requires_grad_(True)
                masked_c_grad['indices_col_fine'] = indices_col_fine
                masked_c_grad['indices_col_mid'] = indices_col_mid
                    
               
                geo_pcl_para_fine = [geo_pcl_grad_fine]
                color_pcl_para_fine = [color_pcl_grad_fine]
                geo_pcl_para_mid = [geo_pcl_grad_mid]
                color_pcl_para_mid = [color_pcl_grad_mid]
                

                masked_c_grad['geo_pcl_grad_fine'] = geo_pcl_grad_fine
                masked_c_grad['color_pcl_grad_fine'] = color_pcl_grad_fine
                masked_c_grad['geo_pcl_grad_mid'] = geo_pcl_grad_mid
                masked_c_grad['color_pcl_grad_mid'] = color_pcl_grad_mid
                
            else:
                masked_c_grad = {}
                #fine level
                geo_pcl_grad_fine = npc_geo_feats_fine.clone().detach().requires_grad_(True)
                color_pcl_grad_fine = npc_col_feats_fine.clone().detach().requires_grad_(True)
                #mid level
                geo_pcl_grad_mid = npc_geo_feats_mid.clone().detach().requires_grad_(True)
                color_pcl_grad_mid = npc_col_feats_mid.clone().detach().requires_grad_(True)
                
                geo_pcl_para_fine = [geo_pcl_grad_fine]
                color_pcl_para_fine = [color_pcl_grad_fine]

                geo_pcl_para_mid = [geo_pcl_grad_mid]
                color_pcl_para_mid = [color_pcl_grad_mid]

                masked_c_grad['geo_pcl_grad_fine'] = geo_pcl_grad_fine
                masked_c_grad['color_pcl_grad_fine'] = color_pcl_grad_fine
                masked_c_grad['geo_pcl_grad_mid'] = geo_pcl_grad_mid
                masked_c_grad['color_pcl_grad_mid'] = color_pcl_grad_mid

        if not self.fix_geo_decoder_mid:
            decoders_para_list += list(
                self.decoders.geo_decoder_mid.parameters())
        if not self.fix_geo_decoder_fine:
            decoders_para_list += list(
                self.decoders.geo_decoder_fine.parameters())
        if not self.fix_color_decoder:
            decoders_para_list += list(
                self.decoders.color_decoder_mid.parameters())
            decoders_para_list += list(
                self.decoders.color_decoder_fine.parameters())

        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]['est_c2w']
                        gt_c2w = keyframe_dict[frame]['gt_c2w']
                    else:
                        c2w = cur_c2w
                        gt_c2w = gt_cur_c2w
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)

        if self.nice:
            # params for different levels
            optim_para_list = [{'params': decoders_para_list, 'lr': 0},
                               {'params': geo_pcl_para_mid, 'lr': 0},
                               {'params': geo_pcl_para_fine, 'lr': 0},
                               {'params': color_pcl_para_mid, 'lr': 0},
                               {'params': color_pcl_para_fine, 'lr': 0},
                               ]
            if self.BA:
                optim_para_list.append({'params': camera_tensor_list, 'lr': 0})
            if self.encode_exposure:
                optim_para_list.append(
                    {'params': self.exposure_feat, 'lr': 0.001})
            optimizer = torch.optim.Adam(optim_para_list)

        if self.more_iters_when_adding:
            if idx > 0 and not color_refine:
                num_joint_iters = np.clip(int(num_joint_iters*frame_pts_add/300), int(
                    self.min_iter_ratio*num_joint_iters), 2*num_joint_iters)
        #num_joint_iters = 300 if not self.more_iters_when_adding
        num_mid_iters =  int(num_joint_iters * self.mid_iter_ratio)
        num_fine_iters =  int(num_joint_iters * (1 - self.mid_iter_ratio))
        for joint_iter in range(num_joint_iters): #300 for mid, 300 for fine
            tic = time.perf_counter()
            if self.frustum_feature_selection and idx > 0:
                geo_feats_fine, col_feats_fine, geo_feats_mid, col_feats_mid = masked_c_grad['geo_pcl_grad_fine'], masked_c_grad['color_pcl_grad_fine'], masked_c_grad['geo_pcl_grad_mid'], masked_c_grad['color_pcl_grad_mid']
                indices_geo_fine = masked_c_grad['indices_geo_fine']
                indices_col_fine = masked_c_grad['indices_col_fine']
                indices_geo_mid = masked_c_grad['indices_geo_mid']
                indices_col_mid = masked_c_grad['indices_col_mid']
                npc_geo_feats_fine[indices_geo_fine] = geo_feats_fine
                npc_col_feats_fine[indices_col_fine] = col_feats_fine
                npc_geo_feats_mid[indices_geo_mid] = geo_feats_mid
                npc_col_feats_mid[indices_col_mid] = col_feats_mid
            else:
                geo_feats_fine, col_feats_fine, geo_feats_mid, col_feats_mid = masked_c_grad['geo_pcl_grad_fine'], masked_c_grad['color_pcl_grad_fine'], masked_c_grad['geo_pcl_grad_mid'], masked_c_grad['color_pcl_grad_mid']
                npc_geo_feats_fine = geo_feats_fine  # all feats
                npc_col_feats_fine = col_feats_fine
                npc_geo_feats_mid = geo_feats_mid
                npc_col_feats_mid = col_feats_mid
            
            '''
            if joint_iter <= (self.geo_iter_first if init else int(num_joint_iters*self.middle_iter_ratio)):
                self.stage = 'geometry_mid'
            elif joint_iter <= int(num_joint_iters*self.fine_iter_ratio):
                self.stage = 'geometry_fine'
            else:
                self.stage = 'color_fine'
            '''
            #self.geo_iter_ratio: 0.4, self.col_iter_ratio: 0.6
            if joint_iter <= (self.geo_iter_first if init else int(num_mid_iters*self.geo_iter_ratio)):
                self.stage = 'geometry_mid'
            elif joint_iter <= int(num_mid_iters):
                self.stage = 'color_mid'
            elif joint_iter <= int(num_mid_iters + num_fine_iters*self.geo_iter_ratio):
                self.stage = 'geometry_fine'
            else:
                self.stage = 'color_fine'
                
            cur_stage = 'init' if init else 'stage'

            optimizer.param_groups[0]['lr'] = cfg['mapping'][cur_stage][self.stage]['decoders_lr']
            optimizer.param_groups[1]['lr'] = cfg['mapping'][cur_stage][self.stage]['geometry_mid_lr']
            optimizer.param_groups[2]['lr'] = cfg['mapping'][cur_stage][self.stage]['geometry_fine_lr']
            if idx == self.n_img-1 and self.color_refine:
                optimizer.param_groups[0]['lr'] = cfg['mapping'][cur_stage]['color_fine']['decoders_lr']
                optimizer.param_groups[1]['lr'] = 0.0
                optimizer.param_groups[2]['lr'] = 0.0
                optimizer.param_groups[3]['lr'] = cfg['mapping'][cur_stage]['color_fine']['color_lr']/10.0
                optimizer.param_groups[4]['lr'] = cfg['mapping'][cur_stage]['color_fine']['color_lr']/10.0
            else:
                optimizer.param_groups[3]['lr'] = cfg['mapping'][cur_stage][self.stage]['color_lr']
                optimizer.param_groups[4]['lr'] = cfg['mapping'][cur_stage][self.stage]['color_lr']

            if self.BA:
                # when to conduct BA
                if joint_iter >= num_mid_iters*(self.geo_iter_ratio+0.2) and (joint_iter <= num_mid_iters*(self.geo_iter_ratio+0.3)):
                    optimizer.param_groups[5]['lr'] = self.BA_cam_lr
                elif joint_iter <= num_mid_iters:
                    optimizer.param_groups[5]['lr'] = 0.0
                elif (joint_iter >= num_mid_iters + num_fine_iters*(self.geo_iter_ratio+0.2)) and (joint_iter <= num_mid_iters + num_fine_iters*(self.geo_iter_ratio+0.3)):
                    optimizer.param_groups[5]['lr'] = self.BA_cam_lr
                else:
                    optimizer.param_groups[5]['lr'] = 0.0


            if self.stage == 'geometry_mid'or self.stage == 'color_mid':
                self.cloud_pos_tensor = torch.tensor(
                        self.npc.cloud_pos(level='mid'), device=self.device)
            elif self.stage == 'geometry_fine'or self.stage == 'color_fine':
                self.cloud_pos_tensor = torch.tensor(
                        self.npc.cloud_pos(level='fine'), device=self.device)
            
            if self.vis_inside:
                self.visualizer.vis(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.npc, self.decoders,
                                    npc_geo_feats_fine, npc_col_feats_fine, freq_override=False, normals=self.cloud_normals_fine,
                                    dynamic_r_query=self.dynamic_r_query['fine'], cloud_pos=self.cloud_pos_tensor,
                                    exposure_feat=self.exposure_feat)
                self.visualizer.vis(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.npc, self.decoders,
                                    npc_geo_feats_mid, npc_col_feats_mid, freq_override=False, normals=self.cloud_normals_mid,
                                    dynamic_r_query=self.dynamic_r_query['mid'], cloud_pos=self.cloud_pos_tensor,                                    exposure_feat=self.exposure_feat)


            optimizer.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []
            batch_r_query_list = []
            batch_r_query_list_mid = []
            batch_r_query_list_fine = []
            exposure_feat_list = []
            indices_tensor = []

            camera_tensor_id = 0
            for frame in optimize_frame:
                if frame != -1:
                    gt_depth = keyframe_dict[frame]['depth'].to(device)
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = keyframe_dict[frame]['est_c2w']

                else:
                    gt_depth = cur_gt_depth.to(device)
                    gt_color = cur_gt_color.to(device)
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = cur_c2w

                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
                    0, H, 0, W, pixs_per_image,
                    H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device, depth_filter=True, return_index=True)
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())

                # which level?

                if self.encode_exposure:
                    self.exposure_feat = self.exposure_feat_shared[0].clone().requires_grad_()
                # print('stage:',self.stage)
                #batch_r_query_list
                if self.use_dynamic_radius:                

                    if frame == -1:
                        batch_r_query_list_mid.append(self.dynamic_r_query['mid'][j, i])
                        batch_r_query_list_fine.append(self.dynamic_r_query['fine'][j, i])
                    else:
                        batch_r_query_list_mid.append(
                            keyframe_dict[frame]['dynamic_r_query_mid'][j, i])
                        batch_r_query_list_fine.append(
                            keyframe_dict[frame]['dynamic_r_query_fine'][j, i])

                if self.encode_exposure:  # needs to render frame by frame
                    exposure_feat_list.append(
                        self.exposure_feat if frame == -1 else keyframe_dict[frame]['exposure_feat'].to(device))
                    # log frame idx of pixels
                    frame_indices = torch.full(
                        (i.shape[0],), frame, dtype=torch.long, device=self.device)
                    indices_tensor.append(frame_indices)

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)

            r_query_list_mid = torch.cat(
                batch_r_query_list_mid) if self.use_dynamic_radius else None
            r_query_list_fine = torch.cat(
                batch_r_query_list_fine) if self.use_dynamic_radius else None
            if self.use_bound:
                with torch.no_grad():
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                    t = (self.bound.unsqueeze(0).to(
                        device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= batch_gt_depth
            else:
                with torch.no_grad():
                    inside_mask = batch_gt_depth <= torch.minimum(
                        10*batch_gt_depth.median(), 1.2*torch.max(batch_gt_depth))
            batch_rays_d, batch_rays_o = batch_rays_d[inside_mask], batch_rays_o[inside_mask]
            batch_gt_depth, batch_gt_color = batch_gt_depth[inside_mask], batch_gt_color[inside_mask]
            if self.use_dynamic_radius:

                r_query_list_mid = r_query_list_mid[inside_mask]
                r_query_list_fine = r_query_list_fine[inside_mask]
 
            geo_feats_dict = {"mid": npc_geo_feats_mid, "fine": npc_geo_feats_fine}
            col_feats_dict = {"mid": npc_col_feats_mid, "fine": npc_col_feats_fine}
            cloud_pos_dict = {"mid": torch.tensor(self.npc.cloud_pos(level='mid'), device=self.device), "fine": torch.tensor(self.npc.cloud_pos(level='fine'), device=self.device)}
            # normals_dict = {"mid": self.cloud_normals_mid, "fine": self.cloud_normals_fine}
            r_query_dict = {"mid": r_query_list_mid, "fine": r_query_list_fine}
            #rendering: use fine level of color feature h
            # ret = self.renderer.render_batch_ray(npc, self.decoders, batch_rays_d, batch_rays_o, device, self.stage,
            #                                      gt_depth=batch_gt_depth, npc_geo_feats=geo_feats_render, #
            #                                     x npc_col_feats=col_feats_render, #
            #                                      is_tracker=True if self.BA else False,
            #                                      cloud_pos=self.cloud_pos_tensor, #
            #                                      normals=cloud_normals_render, #
            #                                      dynamic_r_query=r_query_list, #
            #                                      exposure_feat=None)
            ret = self.renderer.render_batch_ray(npc, self.decoders, batch_rays_d, batch_rays_o, device, self.stage,
                                                 gt_depth=batch_gt_depth, npc_geo_feats=geo_feats_dict, #
                                                 npc_col_feats=col_feats_dict, #
                                                 is_tracker=True if self.BA else False,
                                                 cloud_pos=cloud_pos_dict, #
                                                 normals=self.cloud_normals, #
                                                 dynamic_r_query=r_query_dict.copy(), #
                                                 exposure_feat=None)
            depth, uncertainty, color, valid_ray_mask = ret
            
            depth_mask = (batch_gt_depth > 0) & valid_ray_mask
            depth_mask = depth_mask & (~torch.isnan(depth))
            # calculating geometry loss
            geo_loss = torch.abs(
                batch_gt_depth[depth_mask]-depth[depth_mask]).sum()
            loss = geo_loss.clone()
            if self.stage == 'color_mid':
                if self.encode_exposure:
                    indices_tensor = torch.cat(indices_tensor, dim=0)[
                        inside_mask]
                    start_end = []
                    for i in torch.unique_consecutive(indices_tensor, return_counts=False):
                        match_indices = torch.where(indices_tensor == i)[0]
                        start_idx = match_indices[0]
                        end_idx = match_indices[-1] + 1
                        start_end.append((start_idx.item(), end_idx.item()))
                    for i, exposure_feat in enumerate(exposure_feat_list):
                        start, end = start_end[i]
                        affine_tensor = self.decoders.color_decoder_mid.mlp_exposure(
                            exposure_feat)
                        rot, trans = affine_tensor[:9].reshape(
                            3, 3), affine_tensor[-3:]
                        color_slice = color[start:end].clone()
                        color_slice = torch.matmul(color_slice, rot) + trans
                        color[start:end] = color_slice
                    color = torch.sigmoid(color)
                color_loss = torch.abs(
                    batch_gt_color[depth_mask] - color[depth_mask]).sum()

                weighted_color_loss = self.w_color_loss*color_loss
                loss += weighted_color_loss
            
            elif self.stage == 'color_fine':
                if self.encode_exposure:
                    indices_tensor = torch.cat(indices_tensor, dim=0)[
                        inside_mask]
                    start_end = []
                    for i in torch.unique_consecutive(indices_tensor, return_counts=False):
                        match_indices = torch.where(indices_tensor == i)[0]
                        start_idx = match_indices[0]
                        end_idx = match_indices[-1] + 1
                        start_end.append((start_idx.item(), end_idx.item()))
                    for i, exposure_feat in enumerate(exposure_feat_list):
                        start, end = start_end[i]
                        affine_tensor = self.decoders.color_decoder_fine.mlp_exposure(
                            exposure_feat)
                        rot, trans = affine_tensor[:9].reshape(
                            3, 3), affine_tensor[-3:]
                        color_slice = color[start:end].clone()
                        color_slice = torch.matmul(color_slice, rot) + trans
                        color[start:end] = color_slice
                    color = torch.sigmoid(color)
                color_loss = torch.abs(
                    batch_gt_color[depth_mask] - color[depth_mask]).sum()

                weighted_color_loss = self.w_color_loss*color_loss
                loss += weighted_color_loss

            loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()

            # put selected and updated params back to npc
            if self.frustum_feature_selection:
                geo_feats_fine, geo_feats_mid, col_feats_fine, col_feats_mid = masked_c_grad['geo_pcl_grad_fine'],masked_c_grad['geo_pcl_grad_mid'], masked_c_grad['color_pcl_grad_fine'], masked_c_grad['color_pcl_grad_mid']
                indices_geo_fine = masked_c_grad['indices_geo_fine']
                indices_geo_mid = masked_c_grad['indices_geo_mid']
                indices_col_fine = masked_c_grad['indices_col_fine']
                indices_col_mid = masked_c_grad['indices_col_mid']
                npc_geo_feats_fine,npc_geo_feats_mid, npc_col_feats_fine , npc_col_feats_mid = npc_geo_feats_fine.detach(),npc_geo_feats_mid.detach(), npc_col_feats_fine.detach(), npc_col_feats_mid.detach()
                npc_geo_feats_fine[indices_geo_fine],npc_geo_feats_mid[indices_geo_mid], npc_col_feats_fine[indices_col_fine], npc_col_feats_mid[indices_col_mid] = geo_feats_fine.clone(
                ).detach(), geo_feats_mid.clone().detach(), col_feats_fine.clone().detach(), col_feats_mid.clone().detach()
            else:
                geo_feats_fine, geo_feats_mid, col_feats_fine, col_feats_mid = masked_c_grad['geo_pcl_grad_fine'],masked_c_grad['geo_pcl_grad_mid'], masked_c_grad['color_pcl_grad_fine'], masked_c_grad['color_pcl_grad_mid']
                npc_geo_feats_fine,npc_geo_feats_mid, npc_col_feats_fine, npc_col_feats_mid = geo_feats_fine.detach(), geo_feats_mid.detach(),col_feats_fine.detach(),col_feats_mid.detach()

            toc = time.perf_counter()
            if not self.wandb:
                if self.stage == 'geometry_mid':
                    print('geometry_mid:')
                    print('iter: ', joint_iter, ', time',
                          f'{toc - tic:0.6f}', ', geo_loss: ', f'{geo_loss.item():0.6f}')
                elif self.stage == 'geometry_fine':
                    print('geometry_fine:')
                    print('iter: ', joint_iter, ', time',
                          f'{toc - tic:0.6f}', ', geo_loss: ', f'{geo_loss.item():0.6f}')
                elif self.stage == 'color_mid':
                    print('color_mid:')
                    print('iter: ', joint_iter, ', time', f'{toc - tic:0.6f}',
                          ', geo_loss: ', f'{geo_loss.item():0.6f}', ', color_loss: ', f'{color_loss.item():0.6f}')
                elif self.stage == 'color_fine':
                    print('color_fine:')
                    print('iter: ', joint_iter, ', time', f'{toc - tic:0.6f}',
                          ', geo_loss: ', f'{geo_loss.item():0.6f}', ', color_loss: ', f'{color_loss.item():0.6f}')

            if joint_iter == num_joint_iters-1:
                print('idx: ', idx.item(), ', time', f'{toc - tic:0.6f}', ', geo_loss_pixel: ', f'{(geo_loss.item()/depth_mask.sum().item()):0.6f}',
                      ', color_loss_pixel: ', f'{(color_loss.item()/depth_mask.sum().item()):0.4f}')
                if self.wandb:
                    if not self.gt_camera:
                        wandb.log({'idx_map': int(idx.item()), 'time': float(f'{toc - tic:0.6f}'),
                                   'geo_loss_pixel': float(f'{(geo_loss.item()/depth_mask.sum().item()):0.6f}'),
                                   'color_loss_pixel': float(f'{(color_loss.item()/depth_mask.sum().item()):0.6f}'),
                                   'pts_total_mid': self.npc.index_ntotal('mid'),
                                   'pts_total_fine': self.npc.index_ntotal('fine')})
                    else:
                        wandb.log({'idx': int(idx.item()), 'time': float(f'{toc - tic:0.6f}'),
                                   'geo_loss_pixel': float(f'{(geo_loss.item()/depth_mask.sum().item()):0.6f}'),
                                   'color_loss_pixel': float(f'{(color_loss.item()/depth_mask.sum().item()):0.6f}'),
                                   'pts_total_mid': self.npc.index_ntotal('mid'),
                                   'pts_total_fine': self.npc.index_ntotal('fine')})
                    if self.more_iters_when_adding:
                        wandb.log({'idx_map': int(idx.item()),
                                  'num_joint_iters': num_joint_iters})

        if (not self.vis_inside) or idx == 0:
            self.visualizer.vis(idx, self.num_joint_iters-1, cur_gt_depth, cur_gt_color, cur_c2w, self.npc, self.decoders,
                                geo_feats_dict, col_feats_dict, freq_override=True if idx == 0 else False,
                                # normals=self.cloud_normals_mid, dynamic_r_query=self.dynamic_r_query['mid'],
                                normals=self.cloud_normals, dynamic_r_query=self.dynamic_r_query,
                                cloud_pos=cloud_pos_dict, exposure_feat=self.exposure_feat,
                                cur_total_iters=num_joint_iters, save_rendered_image=True if self.save_rendered_image else False)

        if self.frustum_feature_selection:
            self.npc.update_geo_feats(geo_feats_fine, indices=indices_geo_fine, level='fine')
            self.npc.update_geo_feats(geo_feats_mid, indices=indices_geo_mid, level='mid')
            self.npc.update_col_feats(col_feats_fine, indices=indices_col_fine, level='fine')
            self.npc.update_col_feats(col_feats_mid, indices=indices_col_mid, level='mid')

        else:
            self.npc.update_geo_feats(geo_feats_fine, level='fine')
            self.npc.update_geo_feats(geo_feats_mid, level='mid')
            self.npc.update_col_feats(col_feats_fine, level='fine')
            self.npc.update_col_feats(col_feats_mid, level='mid')

        #update point feature
        self.npc_geo_feats_fine = npc_geo_feats_fine
        self.npc_geo_feats_mid = npc_geo_feats_mid
        self.npc_col_feats_fine = npc_col_feats_fine
        self.npc_col_feats_mid = npc_col_feats_mid
        print('Mapper has updated point features.')

        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.encode_exposure:
            self.exposure_feat_shared[0] = self.exposure_feat.clone().detach()

        if self.BA:
            return cur_c2w
        else:
            return None

    def run(self, npc, time_string):
        setup_seed(1219)
        cfg = self.cfg
        scene_name = cfg['data']['input_folder'].split('/')[-1]
        run_name = cfg['data']['output'].split('/')[-1]

        if self.use_dynamic_radius:
            os.makedirs(f'{self.output}/dynamic_r_frame', exist_ok=True)
        if self.wandb:
            from datetime import datetime
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H%M%S")
            if not self.gt_camera:  # use group to log tracker and mapper
                wandb.init(config=cfg, project=self.project_name, group=f'slam_{scene_name}' if (self.cfg['project_name'] == 'NICER_SLAM_replica' or self.cfg['project_name'] == 'NICER_SLAM_report') else run_name,
                           name='mapper_' +
                           run_name if (self.cfg['project_name'] == 'NICER_SLAM_replica' or self.cfg['project_name']
                                        == 'NICER_SLAM_report') else 'mapper_'+dt_string,
                           settings=wandb.Settings(code_dir="."), dir=self.cfg['wandb_dir'],    # '/cluster/scratch/guohan/point-slam/output'
                           tags=[scene_name])
                wandb.run.log_code(".")
            else:
                wandb.init(config=cfg, project=self.project_name_with_gt, name=run_name,
                           settings=wandb.Settings(code_dir="."), dir=self.cfg['wandb_dir'])    # '/cluster/scratch/guohan/point-slam/output'
                wandb.run.log_code(".")
            wandb.watch((self.decoders.geo_decoder_mid, self.decoders.geo_decoder_fine, 
                        self.decoders.color_decoder_fine), criterion=None, log="all")

        self.npc = npc
        _, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])

        self.estimate_c2w_list[0] = gt_c2w.cpu()
        init = True
        prev_idx = -1
        self.prev_c2w = self.estimate_c2w_list[0]
        while (1):
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1:
                    break
                if self.sync_method == 'strict':
                    if idx % self.every_frame == 0 and idx != prev_idx:
                        break

                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx+self.every_frame//2:
                        break
                elif self.sync_method == 'free':
                    break
                time.sleep(0.1)
            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                prefix = 'Coarse ' if self.coarse_mapper else ''
                print(prefix+"Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx.item()]

            if self.use_dynamic_radius: # generate dynamic radius from radius_add_max. should work on this first for additonal levels
                ratio = self.radius_query_ratio #query ratio, x2 default. 
                intensity = rgb2gray(gt_color.cpu().numpy())
                grad_y = filters.sobel_h(intensity)
                grad_x = filters.sobel_v(intensity)
                color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
                color_grad_mag = np.clip(
                    color_grad_mag, 0.0, self.color_grad_threshold)  # range 0~1
                # mapping based on color gradient

                # hierarchy level dynamic radius
                for level in self.radius_hierarchy.keys():
                    
                    radius_add_max = self.radius_hierarchy[level][list(self.radius_hierarchy[level].keys())[0]]
                    radius_add_min = self.radius_hierarchy[level][list(self.radius_hierarchy[level].keys())[1]]

                    fn_map_r_add = interp1d([0, 0.01, self.color_grad_threshold], [
                                            radius_add_max, radius_add_max, radius_add_min])
                    fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [
                                            ratio*radius_add_max, ratio*radius_add_max, ratio*radius_add_min])
                    dynamic_r_add = fn_map_r_add(color_grad_mag)
                    dynamic_r_query = fn_map_r_query(color_grad_mag)
                 
               
                    self.dynamic_r_add[level], self.dynamic_r_query[level] = torch.from_numpy(dynamic_r_add).to(
                        self.device), torch.from_numpy(dynamic_r_query).to(self.device)
                    # if init:
                    #     torch.save(self.dynamic_r_query[level], f'{self.output}/dynamic_r_frame/r_query_{idx:05d}_{level}.pt')
                        


                    
                    

            color_refine = True if (
                idx == self.n_img-1 and self.color_refine) else False
            if not init:
                lr_factor = cfg['mapping']['lr_factor']
                num_joint_iters = cfg['mapping']['iters']
                self.mapping_window_size = cfg['mapping']['mapping_window_size']*(
                    2 if self.n_img > 4000 else 1)

                if idx == self.n_img-1 and self.color_refine:  # end of SLAM
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    self.geo_iter_ratio = 0.0
                    num_joint_iters *= 10
                    self.fix_color_decoder = True
                    self.frustum_feature_selection = False
                    self.keyframe_selection_method = 'global'
                else:
                    outer_joint_iters = 1

            else:
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = self.iters_first  # more iters on first run

            cur_c2w = self.estimate_c2w_list[idx].to(self.device)
            num_joint_iters = num_joint_iters//outer_joint_iters

            if self.end_correction and idx == self.n_img-1:
                npc_pcl = get_npc_input_pcl(self.npc)
                npc_pcl.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
                npc_pcl.orient_normals_towards_camera_location(
                    camera_location=cur_c2w[:3, 3].cpu().numpy())
                voxel_size = 0.04
                print(npc_pcl)
                num_points = len(npc_pcl.points)
                if num_points > 50000:
                    try:
                        result_fitness, result_ransac = 0, None
                        coef_list = [0.8, 0.9, 0.95]
                        target = npc_pcl.select_by_index(
                            range(int(num_points*0.6)))
                        target_down, target_fpfh = preprocess_point_cloud(
                            target, voxel_size)
                        for coef in coef_list:
                            source = npc_pcl.select_by_index(
                                range(int(num_points*coef), num_points))
                            source_down, source_fpfh = preprocess_point_cloud(
                                source, voxel_size)
                            for i in range(3):  # try 3 times
                                result_ransac_candidate = execute_global_registration(source_down, target_down,
                                                                                      source_fpfh, target_fpfh, voxel_size)
                                if result_ransac_candidate.fitness > result_fitness:
                                    result_ransac = result_ransac_candidate
                                    result_fitness = result_ransac_candidate.fitness
                        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                                         voxel_size, result_ransac.transformation)
                    except Exception as e:
                        print(e)
                    else:
                        if result_icp.fitness > 0.5:  # lower bound
                            print('correction transformation:',
                                  result_icp.transformation)
                            trans_cur = torch.from_numpy(
                                np.copy(result_icp.transformation).astype(np.float32)).to(self.device)
                            cur_c2w_old = cur_c2w.clone()
                            cam_before = get_tensor_from_camera(cur_c2w_old)
                            cur_c2w = torch.matmul(trans_cur, cur_c2w)

                            print(result_icp)
                            cam_after = get_tensor_from_camera(cur_c2w)
                            cam_gt = get_tensor_from_camera(gt_c2w)
                            print(
                                f'End correction, cam_quad_error: {torch.abs(cam_before[:4]-cam_gt[:4]).mean():0.6f} -> {torch.abs(cam_after[:4]-cam_gt[:4]).mean():0.6f}, cam_pos_error: {torch.abs(cam_before[-3:]-cam_gt[-3:]).mean():0.6f} -> {torch.abs(cam_after[-3:]-cam_gt[-3:]).mean():0.6f}')

                            self.estimate_c2w_list[idx] = cur_c2w.cpu()
                            # apply correction
                            index_interval = 1000
                            translation_value = (
                                cur_c2w[:3, 3]-cur_c2w_old[:3, 3]).cpu()
                            print('end translation correction: ',
                                  translation_value)
                            for i in range(idx):
                                if i >= idx-800:
                                    index_diff = abs(i - idx)
                                    decay_factor = np.exp(-index_diff /
                                                          index_interval)
                                    translation_correction = translation_value*decay_factor
                                    self.estimate_c2w_list[i][:3,
                                                              3] += translation_correction
                        else:
                            print(f'result_fitness: {result_icp.fitness}')
                            print('End correction rejected.')
                else:
                    print(f'npc_pts_num: {num_points}')
                    print('End correction rejected.')

            for outer_joint_iter in range(outer_joint_iters):
                # start BA when having enough keyframes
                self.BA = (len(self.keyframe_list) >
                           4) and cfg['mapping']['BA']

                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, None,
                                      self.keyframe_dict, self.keyframe_list, cur_c2w, color_refine=color_refine)
                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

            if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) and (idx not in self.keyframe_list) and (not torch.isinf(gt_c2w).any()) and (not torch.isnan(gt_c2w).any()):
                self.keyframe_list.append(idx)
                dic_of_cur_frame = {'gt_c2w': gt_c2w.detach().cpu(), 'idx': idx, 'color': gt_color.detach().cpu(),
                                    'depth': gt_depth.detach().cpu(), 'est_c2w': cur_c2w.clone().detach()}
                if self.use_dynamic_radius:
                    dic_of_cur_frame.update(
                        {'dynamic_r_query_mid': self.dynamic_r_query['mid'].detach()})
                    dic_of_cur_frame.update(
                        {'dynamic_r_query_fine': self.dynamic_r_query['fine'].detach()})
                if self.encode_exposure:
                    dic_of_cur_frame.update(
                        {'exposure_feat': self.exposure_feat.detach().cpu()})
                self.keyframe_dict.append(dic_of_cur_frame)

            init = False
            self.prev_c2w = self.estimate_c2w_list[idx]

            if (idx % 300 == 0 or idx == self.n_img-1):
                cloud_pos = np.array(self.npc.input_pos())
                cloud_rgb = np.array(self.npc.input_rgb())
                point_cloud = np.hstack((cloud_pos, cloud_rgb))
                npc_cloud_fine = np.array(self.npc.cloud_pos('fine'))
                npc_cloud_mid = np.array(self.npc.cloud_pos('mid'))
                if idx == self.n_img-1:
                    np.save(f'{self.output}/final_point_cloud', point_cloud)
                    np.save(f'{self.output}/npc_cloud_fine', npc_cloud_fine)
                    np.save(f'{self.output}/npc_cloud_mid', npc_cloud_mid)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud_pos)
                    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb/255.0)
                    o3d.io.write_point_cloud(
                        f'{self.output}/final_point_cloud.ply', pcd)
                    print('Saved point cloud and point normals.')
                if self.wandb:
                    pc_list = []
                    pc_list.append(wandb.Object3D(point_cloud, caption="input"))
                    pc_list.append(wandb.Object3D(npc_cloud_fine, caption="fine"))
                    pc_list.append(wandb.Object3D(npc_cloud_mid, caption="mid"))
                    wandb.log(
                        {f'Cloud/point_cloud_{idx:05d}': pc_list})
                        # {f'Cloud/point_cloud_{idx:05d}': wandb.Object3D(point_cloud)})

            if (idx > 0 and idx % self.ckpt_freq == 0) or idx == self.n_img-1:
                self.logger.log(idx, self.keyframe_dict, self.keyframe_list,
                                selected_keyframes=self.selected_keyframes
                                if self.save_selected_keyframes_info else None, npc=self.npc)

            # mapping of first frame is done, can begin tracking
            self.mapping_first_frame[0] = 1
            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if idx == self.n_img-1:
                print('Color refinement done.')
                print('Mapper finished.')
                break

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

        try:
            print('✨ Point-SLAM finished, evaluating...')
            ate_rmse = subprocess.check_output(['python', '-u', 'src/tools/eval_ate.py', str(cfg['config_path']), '--output', str(cfg['data']['output'])],
                                               text=True, stderr=subprocess.STDOUT)
            print('ate_rmse: ', ate_rmse)
            ate_rmse = literal_eval(str(ate_rmse))

            ate_rmse_no_align = subprocess.check_output(['python', '-u', 'src/tools/eval_ate.py', str(cfg['config_path']), '--output', str(cfg['data']['output']), '--no_align'],
                                                        text=True, stderr=subprocess.STDOUT)
            print('ate_rmse_wo_align: ', ate_rmse_no_align)
            ate_rmse_no_align = literal_eval(str(ate_rmse_no_align))

            if self.wandb:
                wandb.log(
                    {'ate-rmse': ate_rmse["absolute_translational_error.rmse"]})
                wandb.log(
                    {'ate-rmse-no-align': ate_rmse_no_align["absolute_translational_error.rmse"]})
            print('Successfully evaluated trajectory.')
        except Exception as e:
            traceback.print_exception(e)
            self.save_ckpts = True  # in case needed
            print('Failed to evaluate trajectory.')

        # re-render frames at the end for meshing
        # if cfg['dataset'] == 'replica':
        #     print('Starting re-rendering frames...')
        #     render_idx, frame_cnt, psnr_sum, ssim_sum, lpips_sum, depth_l1_render = 0, 0, 0, 0, 0, 0
        #     os.makedirs(f'{self.output}/rendered_every_frame', exist_ok=True)
        #     os.makedirs(f'{self.output}/rendered_image', exist_ok=True)
        #     if self.eval_img:
        #         cal_lpips = LearnedPerceptualImagePatchSimilarity(
        #             net_type='alex', normalize=True).to(self.device)  
        #     try:
        #         while render_idx < self.n_img:
        #             _, gt_color, gt_depth, gt_c2w = self.frame_reader[render_idx]
        #             cur_c2w = self.estimate_c2w_list[render_idx].to(
        #                 self.device)
        #             npc_geo_feats = dict.fromkeys(self.radius_hierarchy.keys(), {})
        #             npc_col_feats = dict.fromkeys(self.radius_hierarchy.keys(), {})
        #             cloud_pos_dict = dict.fromkeys(self.radius_hierarchy.keys(), {})
        #             r_query_frame = dict.fromkeys(self.radius_hierarchy.keys(), {})
        #             if self.use_dynamic_radius:
        #                 for key in self.radius_hierarchy.keys():
        #                     npc_geo_feats[key] = self.npc.get_geo_feats(key)
        #                     npc_col_feats[key] = self.npc.get_col_feats(key)
        #                     cloud_pos_dict[key] = torch.tensor(self.npc.cloud_pos(level=key))
        #                     r_query_frame[key] = torch.load(f'{self.output}/dynamic_r_frame/r_query_{render_idx:05d}_{key}.pt', map_location=self.device)
                                                          
                                                          
        #             cur_frame_depth, cur_frame_color = self.visualizer.vis_value_only(idx, 0, gt_depth, gt_color, cur_c2w, self.npc, self.decoders,
        #                                                                               npc_geo_feats, npc_col_feats, freq_override=True, normals=self.cloud_normals,
        #                                                                               dynamic_r_query=r_query_frame, cloud_pos=self.cloud_pos_tensor, exposure_feat=None)
        #             np.save(f'{self.output}/rendered_every_frame/depth_{render_idx:05d}',
        #                     cur_frame_depth.cpu().numpy())
        #             np.save(f'{self.output}/rendered_every_frame/color_{render_idx:05d}',
        #                     cur_frame_color.cpu().numpy())
        #             if render_idx % 5 == 0:
        #                 img = cv2.cvtColor(
        #                     cur_frame_color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
        #                 cv2.imwrite(os.path.join(
        #                     f'{self.output}/rendered_image', f'frame_{render_idx:05d}.png'), img)
        #             if self.wandb and self.eval_img:
        #                 mse_loss = torch.nn.functional.mse_loss(
        #                     gt_color[gt_depth > 0], cur_frame_color[gt_depth > 0])
        #                 psnr_frame = -10. * torch.log10(mse_loss)
        #                 ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), cur_frame_color.transpose(0, 2).unsqueeze(0).float(),
        #                                      data_range=1.0, size_average=True)
        #                 lpips_value = cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
        #                                         torch.clamp(cur_frame_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0)).item()
        #                 psnr_sum += psnr_frame
        #                 ssim_sum += ssim_value
        #                 lpips_sum += lpips_value
        #                 wandb.log({'idx_frame': render_idx,
        #                           'psnr_frame': psnr_frame})
        #             depth_l1_render += torch.abs(
        #                 gt_depth[gt_depth > 0] - cur_frame_depth[gt_depth > 0]).mean().item()
        #             render_idx += cfg['mapping']['every_frame']
        #             frame_cnt += 1
        #             if render_idx % 400 == 0:
        #                 print(f'frame {render_idx}')
        #         else:
        #             if self.wandb and self.eval_img:
        #                 avg_psnr = psnr_sum / frame_cnt
        #                 avg_ssim = ssim_sum / frame_cnt
        #                 avg_lpips = lpips_sum / frame_cnt
        #                 wandb.log({'avg_ms_ssim': avg_ssim})
        #                 wandb.log({'avg_psnr': avg_psnr})
        #                 wandb.log({'avg_lpips': avg_lpips})
        #             print(f'depth_l1_render: {depth_l1_render/frame_cnt}')
        #             if self.wandb:
        #                 wandb.log(
        #                     {'depth_l1_render': depth_l1_render/frame_cnt})
        #     except Exception as e:
        #         traceback.print_exception(e)
        #         print('Re-rendering frames failed.')
        #     print(f'Finished rendering {frame_cnt} frames.')

        # if cfg['dataset'] == 'replica':
        #     if cfg['meshing']['eval_rec']:
        #         try:
        #             print('Evaluating reconstruction...')
        #             params_list = ['python', '-u', 'src/tools/get_mesh_tsdf_fusion.py',
        #                            str(cfg['config_path']
        #                                ), '--input_folder', cfg['data']['input_folder'],
        #                            '--output', cfg['data']['output'], '--no_render']
        #             if cfg['dataset'] != 'replica':
        #                 params_list.append('--no_eval')

        #             try:
        #                 result_recon_obj = subprocess.run(
        #                     params_list, text=True, check=True, capture_output=True)
        #                 result_recon = str(result_recon_obj.stdout)
        #             except subprocess.CalledProcessError as e:
        #                 print(e.stderr)

        #             if cfg['dataset'] == 'replica':
        #                 # requires only one pair {} inside the printed result
        #                 print("--result_recon--")
        #                 print(result_recon)
        #                 print("--end of result_recon--")
        #                 start_index = result_recon.find('{')
        #                 end_index = result_recon.find('}')
        #                 print("--result_dict--")
        #                 print(start_index, ' ', end_index+1)
        #                 result_dict = result_recon[start_index:end_index+1]
        #                 print(result_dict)
        #                 print("--end of result_dict--")
        #                 result_dict = literal_eval(result_dict)
        #                 if self.wandb:
        #                     wandb.log(result_dict)
        #             torch.cuda.empty_cache()

        #         except Exception as e:
        #             traceback.print_exception(e)
        #             print('Failed to evaluate 3D reconstruction.')

        if os.path.exists(f'{self.output}/dynamic_r_frame'):
            shutil.rmtree(f'{self.output}/dynamic_r_frame')
        if os.path.exists(f'{self.output}/rendered_every_frame'):
            shutil.rmtree(f'{self.output}/rendered_every_frame')
        if not self.save_ckpts:
            if os.path.exists(f'{self.output}/ckpts'):
                shutil.rmtree(f'{self.output}/ckpts')
        if self.wandb:
            print('wandb finished.')
            wandb.finish()