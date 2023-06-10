import torch
import numpy as np
import numpy.ma as ma
import warnings
import random
import open3d as o3d

import faiss
import faiss.contrib.torch_utils
from src.common import setup_seed, clone_kf_dict



class NeuralPointCloud(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.c_dim = cfg['model']['c_dim']
        self.device = cfg['mapping']['device']
        self.cuda_id = 0
        self.pts_along_ray = cfg['pts_along_ray']   # deprecated
        self.use_normals = cfg['use_normals']
        self.use_dynamic_radius = cfg['use_dynamic_radius']

        self.search_method = cfg['pointcloud']['nn_search_method']
        self.nn_num = cfg['pointcloud']['nn_num']

        self.nlist = cfg['pointcloud']['nlist']
        self.radius_add = cfg['pointcloud']['radius_add']
        self.radius_min = cfg['pointcloud']['radius_min']
        self.radius_query = cfg['pointcloud']['radius_query']
        self.radius_mesh = cfg['pointcloud']['radius_mesh']
        self.add_along_normals = cfg['pointcloud']['add_along_normals']
        self.fix_interval_when_add_along_ray = cfg['pointcloud']['fix_interval_when_add_along_ray']

        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_add = cfg['pointcloud']['N_add']
        self.near_end_surface = cfg['pointcloud']['near_end_surface']
        self.far_end_surface = cfg['pointcloud']['far_end_surface']

        # Hierarchical point cloud are stored as dictionary
        self.radius_hierarchy = cfg['pointcloud']['radius_hierarchy']
        radius_levels = self.radius_hierarchy.keys()
        self._cloud = {}
        self._cloud_idx0 = []
        
        self._cloud_pos = {}# (input_pos) * N_add
        self._cloud_pos_idx0 = []

        self._cloud_normal = {}
        self._cloud_normal_idx0_fine = []
        self._cloud_normal_idx0_mid = []

        self._input_pos = []  # to save locations of the rgbd input
        self._input_rgb = []
        self._input_normal = []
        self._input_normal_cartesian = []

        self._pts_num = {}
        self.geo_feats = {}
        self.col_feats = {}
        
        self.keyframe_dict = []

        self.resource = faiss.StandardGpuResources()
        self.index = {}
        self.nprobe = cfg['pointcloud']['nprobe']
        
        setup_seed(1219)

    def cloud(self, level=None, index=None):
        if index is None:
            return self._cloud
        return self._cloud[level][index]

    def append_cloud(self, value, level=None):
        self._cloud[level].append(value)

    def cloud_pos(self, level, index=None):
        

        if index is None:
            return self._cloud_pos[level]
        return self._cloud_pos[level][index]


    def input_pos(self):
        return self._input_pos

    def input_rgb(self):
        return self._input_rgb

    def input_normal(self):
        return self._input_normal

    def input_normal_cartesian(self):
        return self._input_normal_cartesian

    def cloud_normal(self,level=None):
        
            return self._cloud_normal[level]

    def append_cloud_pos(self, value, level=None):
        self._cloud_pos[level].append(value)

    def pts_num(self):
        return self._pts_num

    def add_pts_num(self):
        self._pts_num += 1

    def set_pts_num(self, value):
        self._pts_num = value

    def set_keyframe_dict(self, value):
        self.keyframe_dict = value

    def get_keyframe_dict(self):
        return clone_kf_dict(self.keyframe_dict)

    def get_index(self,level=None):
        return self.index[level]

    def index_is_trained(self):
        return self.index.is_trained

    def index_train(self, xb):
        assert torch.is_tensor(xb), 'use tensor to train FAISS index'
        self.index.train(xb)
        return self.index.is_trained

    def index_ntotal(self, level):
        return self.index[level].ntotal

    def index_set_nprobe(self, value):
        self.index.nprobe = value

    def index_get_nprobe(self):
        return self.index.nprobe

    def get_device(self):
        return self.device

    def get_c_dim(self):
        return self.c_dim

    def get_radius_query(self):
        return self.radius_query

    def get_radius_add(self):
        return self.radius_add

    def get_geo_feats(self, level=None):
        return self.geo_feats[level]
    
    def get_col_feats(self, level=None):
        return self.col_feats[level]
    

    def update_geo_feats(self, feats, indices=None,level=None):
        #update geometric feature of required level
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self.geo_feats[level][indices] = feats.clone().detach()
        else:
            assert feats.shape[0] == self.geo_feats[level].shape[0], 'feature shape[0] mismatch'
            self.geo_feats[level] = feats.clone().detach()
            
        


    def update_col_feats(self, feats, indices=None, level=None):
        #update color feature of required level
        assert torch.is_tensor(feats), 'use tensor to update features'
        if indices is not None:
            self.col_feats[level][indices] = feats.clone().detach()
        else:
            assert feats.shape[0] == self.col_feats[level].shape[0], 'feature shape[0] mismatch'
            self.col_feats[level] = feats.clone().detach()

    def cart2sph(self, xyz):
        # transform normals from cartesian to sphere angles
        # xyz should be tensor of N*3, and normalized
        normals_sph = torch.zeros(xyz.shape[0], 2, device=xyz.device)
        xy = xyz[:, 0]**2 + xyz[:, 1]**2
        normals_sph[:, 0] = torch.atan2(torch.sqrt(xy), xyz[:, 2])
        normals_sph[:, 1] = torch.atan2(xyz[:, 1], xyz[:, 0])
        return normals_sph

        
                
                
    def add_neural_points(self, batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color,
                          normals=None, train=False, is_pts_grad=False, dynamic_radius=None, level=None,idx=None):
        """
        Add multiple neural points, will use depth filter when getting these samples.
        Add point to all levels of point cloud.

        Args: 
            batch_rays_o (tensor): (N,3)
            batch_rays_d (tensor): (N,3)
            batch_gt_depth (tensor): (N,)
            batch_gt_color (tensor): (N,3)
            normals (tensor, optional): Defaults to None.
        """
        if idx == 0:
            # initialize point cloud hierarchy
            self._cloud_pos.setdefault(level, [])
            self._cloud_normal.setdefault(level, [])
            self._pts_num.setdefault(level, 0)
            self.index.setdefault(level, faiss.index_cpu_to_gpu(self.resource,
                                           self.cuda_id,
                                           faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, self.nlist, faiss.METRIC_L2)))
            self.index[level].nprobe = self.nprobe
            # initialize geometry and color features for different hierarchy levels
            self.geo_feats.setdefault(level, None)
            self.col_feats.setdefault(level, None)
            
            
        if normals is not None:
            assert batch_rays_o.shape[0] == normals.shape[0], "Different input size for point positions and normals"

        if batch_rays_o.shape[0]:
            mask = batch_gt_depth > 0
            batch_gt_color = batch_gt_color*255
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = \
                batch_rays_o[mask], batch_rays_d[mask], batch_gt_depth[mask], batch_gt_color[mask]

            pts_gt = batch_rays_o[..., None, :] + batch_rays_d[...,
                                                               None, :] * batch_gt_depth[..., None, None]
            mask = torch.ones(pts_gt.shape[0], device=self.device).bool()
            pts_gt = pts_gt.reshape(-1, 3)
            self._input_pos.extend(pts_gt[mask].tolist())
            self._input_rgb.extend(batch_gt_color[mask].tolist())

            if normals is not None:
                normals_sph = self.cart2sph(normals[mask])
                self._input_normal.extend(normals_sph.tolist())
                self._input_normal_cartesian.extend(normals[mask].tolist())

            gt_depth_surface = batch_gt_depth.unsqueeze(
                -1).repeat(1, self.N_add)
            t_vals_surface = torch.linspace(
                0.0, 1.0, steps=self.N_add, device=self.device)
            if self.N_add > 1:
                if self.add_along_normals:
                    z_vals = batch_gt_depth.unsqueeze(-1).repeat(1, 1)
                elif self.fix_interval_when_add_along_ray:
                    # add along ray, interval unrelated to depth
                    intervals = torch.linspace(-0.04, 0.04, steps=self.N_add,
                                            device=self.device).unsqueeze(0)
                    z_vals = gt_depth_surface + intervals
                else:  # add along ray, interval related to depth
                    z_vals_surface = self.near_end_surface*gt_depth_surface * (1.-t_vals_surface) + \
                        self.far_end_surface * \
                        gt_depth_surface * (t_vals_surface)
                    z_vals = z_vals_surface
            else:  # pts only at rgbd input
                z_vals = batch_gt_depth.unsqueeze(-1).repeat(1, 1)
            
            
            if self.index[level].is_trained:
                
                # search neighbors within the dynamic radii. Different radius settings for different levels
                _, _, neighbor_num_gt = self.find_neighbors_faiss(
                    pts_gt, step='add', is_pts_grad=is_pts_grad, dynamic_radius=dynamic_radius,level=level)
                
                mask = (neighbor_num_gt == 0) 
                    
            
            if normals is not None:
                
                if self.add_along_normals:
                    self._cloud_normal[level].extend(
                        normals_sph.repeat(self.N_add, 1).tolist())
                else:
                    self._cloud_normal[level].extend(
                        normals_sph.repeat_interleave(self.N_add, dim=0).tolist())
            
            
            
            if not self.add_along_normals:
                pts = batch_rays_o[..., None, :] + \
                batch_rays_d[..., None, :] * z_vals[..., :, None]
                
                pts = pts[mask]  # use mask from pts_gt for auxiliary points         

                pts = pts.reshape(-1, 3)


            else:
                intervals = torch.linspace(-0.04, 0.04, steps=self.N_add,
                                        device=self.device).unsqueeze(-1).unsqueeze(-1)

                intervals = intervals.expand(self.N_add, mask.sum(), 3)
                pts = intervals * normals[mask] + pts_gt[mask]
                pts = pts.reshape(-1, 3)
                
            
            # add points to corresponding level   
            self._cloud_pos[level] += pts.tolist()
            self._pts_num[level] += pts.shape[0]
            
            # feature concatenation and initialization for current frame
            if self.geo_feats[level] is None:
                self.geo_feats[level] = torch.zeros(
                    [self._pts_num[level], self.c_dim], device=self.device).normal_(mean=0, std=0.1)

                self.col_feats[level] = torch.zeros(
                    [self._pts_num[level], self.c_dim], device=self.device).normal_(mean=0, std=0.1)
            
            else:
                self.geo_feats[level] = torch.cat([self.geo_feats[level],
                                    torch.zeros([pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)], 0)
    
                self.col_feats[level] = torch.cat([self.col_feats[level],
                                    torch.zeros([pts.shape[0], self.c_dim], device=self.device).normal_(mean=0, std=0.1)], 0)
                

            
   
            if train or not self.index[level].is_trained:
                self.index[level].train(pts)
            
            self.index[level].train(torch.tensor(self._cloud_pos[level], device=self.device))
                
            self.index[level].add(pts)

      
            
            
            return torch.sum(mask)
        else:
            return 0
        
 
        
    def update_points(self, idxs, geo_feats, col_feats, detach=True): # NOT USED?
        """
        Update point cloud features.

        Args:
            hierarchy_level(list): 
            idxs (list): 
            geo_feats (list of tensors): 
            col_feats (list of tensors): 
        """
        assert len(geo_feats) == len(col_feats), "feature size mismatch"
        for _, idx in enumerate(idxs):
            for level in self.radius_hierarchy.keys():
                if detach:
                    self._cloud[level][idx].geo_feat = geo_feats[_].clone().detach()
                    self._cloud[level][idx].col_feat = col_feats[_].clone().detach()
                else:
                    self._cloud[level][idx].geo_feat = geo_feats[_]
                    self._cloud[level][idx].col_feat = col_feats[_]
        geo_feats = torch.cat(geo_feats, 0)
        col_feats = torch.cat(col_feats, 0)
        if detach:
            self.geo_feats[idxs] = geo_feats.clone().detach()
            self.col_feats[idxs] = col_feats.clone().detach()
        else:
            self.geo_feats[idxs] = geo_feats
            self.col_feats[idxs] = col_feats

    def find_neighbors_faiss(self, pos, step='add', retrain=False, is_pts_grad=False, dynamic_radius=None,level=None):
        """
        Query neighbors using faiss.

        Args:
            pos (tensor): points to find neighbors
            step (str): 'add'|'query'
            retrain (bool, optional): if to retrain the index cluster of IVF
            is_pts_grad: whether it's the points choosen based on color grad, will use smaller radius when looking for neighbors
            dynamic_radius (tensor, optional): choose every radius differently based on its color gradient

        Returns:
            indices: list of variable length list of neighbors index, [] if none
        """
        if (not self.index[level].is_trained) or retrain:
            self.index_fine.train(self._cloud_pos[level])
 


        assert step in ['add', 'query', 'mesh']
        split_pos = torch.split(pos, 65000, dim=0)
        D_list = []
        I_list = []
        for split_p in split_pos:

            
    
            D, I = self.index[level].search(split_p.float(), self.nn_num)
                
            D_list.append(D)
            I_list.append(I)
        D = torch.cat(D_list, dim=0)
        I = torch.cat(I_list, dim=0)

        if step == 'query':
            radius = self.radius_query
        elif step == 'add':
            if not is_pts_grad:
                radius = self.radius_add
            else:
                radius = self.radius_min
        else:
            radius = self.radius_mesh

        if dynamic_radius is not None:
            assert pos.shape[0] == dynamic_radius.shape[0], 'shape mis-match for input points and dynamic radius'
            neighbor_num = (D < dynamic_radius.reshape(-1, 1)
                            ** 2).sum(axis=-1).int()
        else:
            neighbor_num = (D < radius**2).sum(axis=-1).int()

        return D, I, neighbor_num

    def merge_points(self, keyframe_list):
        pass

    def get_feature_at_pos(self, p, feat_name, hierarchy_level):  # not used, use this in decoder
        if torch.is_tensor(p):
            p = p.detach().cpu().numpy().reshape((-1, 3)).astype(np.float32)
        else:
            p = np.asarray(p).reshape((-1, 3)).astype(np.float32)
        D, I, neighbor_num = self.find_neighbors_faiss(p, step='query')
        D, I, neighbor_num = [torch.from_numpy(i).to(
            self.device) for i in (D, I, neighbor_num)]

        c = torch.zeros([p.shape[0], self.c_dim],
                        device=self.device).normal_(mean=0, std=0.01)
        has_neighbors = neighbor_num > 0

        c_temp = torch.cat([torch.sum(torch.cat([getattr(self._cloud[hierarchy_level][I[i, j].item()], feat_name) / (torch.sqrt(D[i, j])+1e-10)
                                                 for j in range(neighbor_num[i].item())], 0), 0) / torch.sum(1.0 / (torch.sqrt(D[i, :neighbor_num[i]])+1e-10))
                            for i in range(p.shape[0]) if neighbor_num[i].item() > 0], dim=0)
        c_temp = c_temp.reshape(-1, self.c_dim)
        c[has_neighbors] = c_temp

        return c, has_neighbors

    def sample_near_pcl(self, rays_o, rays_d, near, far, num,dynamic_r_query=None, level=None):
        """
        For pixels with 0 depth readings, preferably sample near point cloud.

        Args:
            rays_o (tensor): _description_
            rays_d (tensor): _description_
            near : near end for sampling along this ray
            far: far end
            num (int): stratified sampling num between near and far
        """
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        n_rays = rays_d.shape[0]
        intervals = 25
        z_vals = torch.linspace(near, far, steps=intervals, device=self.device)
        pts = rays_o[..., None, :] + \
            rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(-1, 3)

        if torch.is_tensor(far):
            far = far.item()
        z_vals_section = np.linspace(near, far, intervals)
        z_vals_np = np.linspace(near, far, num)
        z_vals_total = np.tile(z_vals_np, (n_rays, 1))

        pts_split = torch.split(pts, 65000)  # limited by faiss bug
        Ds, Is, neighbor_nums = [], [], []
        for pts_batch in pts_split:
            D, I, neighbor_num = self.find_neighbors_faiss(
                pts_batch, step='query',dynamic_radius=dynamic_r_query,level=level)
            D, I, neighbor_num = D.cpu().numpy(), I.cpu().numpy(), neighbor_num.cpu().numpy()
            Ds.append(D)
            Is.append(I)
            neighbor_nums.append(neighbor_num)
        D = np.concatenate(Ds, axis=0)
        I = np.concatenate(Is, axis=0)
        neighbor_num = np.concatenate(neighbor_nums, axis=0)

        neighbor_num = neighbor_num.reshape((n_rays, -1))
        neighbor_num_bool = neighbor_num.reshape((n_rays, -1)).astype(bool)
        invalid = neighbor_num_bool.sum(axis=-1) < 2

        if invalid.sum(axis=-1) < n_rays:
            r, c = np.where(neighbor_num[~invalid].astype(bool))
            idx = np.concatenate(
                ([0], np.flatnonzero(r[1:] != r[:-1])+1, [r.size]))
            out = [c[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
            z_vals_valid = np.asarray([np.linspace(
                z_vals_section[item[0]], z_vals_section[item[1]], num=num) for item in out])
            z_vals_total[~invalid] = z_vals_valid

        invalid_mask = torch.from_numpy(invalid).to(self.device)
        return torch.from_numpy(z_vals_total).float().to(self.device), invalid_mask


class NeuralPoint(object):
    def __init__(self, idx, pos, c_dim, normal=None, device=None):
        """
        Init a neural point
        Args:
            idx (int): _description_
            pos (tensor): _description_
            c_dim (int): _description_
            normal (_type_, optional): _description_. Defaults to None.
            device (_type_, optional): _description_. Defaults to None.
        """
        assert torch.is_tensor(pos)
        self.position = pos.to(device=device)
        self.idx = idx
        if normal is not None:
            self.normal = torch.tensor(normal).to(device)
        self.geo_feat = torch.zeros(
            [1, c_dim], device=device).normal_(mean=0, std=0.01)
        self.col_feat = torch.zeros(
            [1, c_dim], device=device).normal_(mean=0, std=0.01)
