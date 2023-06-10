import os
import time
import types

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager, NamespaceProxy

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Renderer import Renderer
from src.neural_point import NeuralPointCloud
from src.common import setup_seed

torch.multiprocessing.set_sharing_strategy('file_system')


class NeuralPointCloudProxy(NamespaceProxy):
    _exposed_ = tuple(dir(NeuralPointCloud))

    def __getattr__(self, name):
        result = super().__getattr__(name)
        try:
            if isinstance(result, types.MethodType):
                def wrapper(*args, **kwargs):
                    return self._callmethod(name, args, kwargs)
                return wrapper
        except:
            pass
        return result


class Point_SLAM():
    """
    NICER_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args, share_npc=True, share_decoders=True):

        self.cfg = cfg
        self.args = args
        self.nice = True

        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            cfg['data']['output'] = args.output
            self.output = args.output

        if args.gt_camera:
            cfg['tracking']['gt_camera'] = True
        if args.depth_limit:
            cfg['tracking']['depth_limit'] = True
        elif args.no_depth_limit:
            cfg['tracking']['depth_limit'] = False
        if args.kf_selection:
            cfg['mapping']['use_kf_selection'] = True
        if args.wandb:
            cfg['wandb'] = True
        elif args.no_wandb:
            cfg['wandb'] = False
        if args.dynamic_r:
            cfg['use_dynamic_radius'] = True
        elif args.fixed_r:
            cfg['use_dynamic_radius'] = False
        if args.use_viewdir:
            cfg['use_view_direction'] = True
        elif args.no_viewdir:
            cfg['use_view_direction'] = False
        if args.encode_viewdir:
            cfg['model']['encode_viewd'] = True
        elif args.no_encode_viewdir:
            cfg['model']['encode_viewd'] = False
        if args.use_exposure:
            cfg['model']['encode_exposure'] = True
        elif args.no_exposure:
            cfg['model']['encode_exposure'] = False
        if args.end_correct:
            cfg['mapping']['end_correction'] = True
        elif args.no_end_correct:
            cfg['mapping']['end_correction'] = False
        if args.use_color_track:
            cfg['tracking']['use_color_in_tracking'] = True
        elif args.no_color_track:
            cfg['tracking']['use_color_in_tracking'] = False
        if args.use_BA:
            cfg['mapping']['BA'] = True
        elif args.no_BA:
            cfg['mapping']['BA'] = False
        if args.eval_img:
            cfg['rendering']['eval_img'] = True
        elif args.no_eval_img:
            cfg['rendering']['eval_img'] = False
        if args.rel_pos_in_col:
            cfg['model']['encode_rel_pos_in_col'] = True
        elif args.no_rel_pos_in_col:
            cfg['model']['encode_rel_pos_in_col'] = False
        if args.track_color:
            cfg['tracking']['sample_with_color_grad'] = True
        elif args.track_uniform:
            cfg['tracking']['sample_with_color_grad'] = False
        if args.radius_add_max is not None:
            cfg['pointcloud']['radius_add_max'] = args.radius_add_max
        if args.radius_add is not None:
            cfg['pointcloud']['radius_add'] = args.radius_add
        if args.radius_query is not None:
            cfg['pointcloud']['radius_query'] = args.radius_query
        if args.track_w_color_loss is not None:
            cfg['tracking']['w_color_loss'] = args.track_w_color_loss
        if args.track_iter is not None:
            cfg['tracking']['iters'] = args.track_iter
        if args.map_iter is not None:
            cfg['mapping']['iters'] = args.map_iter
        if args.min_iter_ratio is not None:
            cfg['mapping']['min_iter_ratio'] = args.min_iter_ratio
        if args.map_every is not None:
            cfg['mapping']['every_frame'] = args.map_every
        if args.kf_every is not None:
            cfg['mapping']['keyframe_every'] = args.kf_every
        if args.map_win_size is not None:
            cfg['mapping']['mapping_window_size'] = args.map_win_size
        if args.kf_t_thre is not None:
            cfg['mapping']['kf_trans_thre'] = args.kf_t_thre
        if args.kf_r_thre is not None:
            cfg['mapping']['kf_rot_thre'] = args.kf_r_thre
        if args.project_name is not None:
            cfg['project_name'] = args.project_name
        self.cfg = cfg

        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        if cfg['mapping']['save_rendered_image']:
            os.makedirs(f'{self.output}/rendered_image', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        model = config.get_model(cfg,  nice=self.nice)
        self.shared_decoders = model

        self.scale = cfg['scale']
        self.load_pretrain(cfg)

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()
        self.mapping_cnt.share_memory_()
        self.exposure_feat = torch.zeros((1, cfg['model']['exposure_dim'])).normal_(
            mean=0, std=0.01).to(self.cfg['mapping']['device'])
        self.exposure_feat.share_memory_()
        if share_decoders:
            self.shared_decoders = self.shared_decoders.to(
                self.cfg['mapping']['device'])
            self.shared_decoders.share_memory()

        if share_npc:
            BaseManager.register('NeuralPointCloud', NeuralPointCloud)
            manager = BaseManager()
            manager.start()
            self.npc = manager.NeuralPointCloud(cfg)
        else:
            self.npc = NeuralPointCloud(cfg)

        self.renderer = Renderer(cfg, args, self)
        self.renderer_map = Renderer(cfg, args, self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print("")
        print(f"⭐️ INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"⭐️ INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"⭐️ INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"⭐️ INFO: The mesh can be found under {self.output}/mesh/")
        print(
            f"⭐️ INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
        self.shared_decoders.geo_decoder_mid.load_state_dict(
            middle_dict, strict=False)
        self.shared_decoders.geo_decoder_fine.load_state_dict(
            middle_dict, strict=False)

    def tracking(self, rank, npc, time_string):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        while (1):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run(npc, time_string)

    def mapping(self, rank, npc, time_string):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run(npc, time_string)

    def coarse_mapping(self, rank):
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run()

    def run(self):
        """
        Dispatch Threads. # this func, when called, act as main process
        """
        setup_seed(1219)
        from datetime import datetime
        now = datetime.now()
        time_string = now.strftime("%Y%m%d_%H%M%S")

        processes = []
        for rank in range(3):
            if rank == 0:
                p = mp.Process(name='tracker', target=self.tracking,
                               args=(rank, self.npc, time_string))
            elif rank == 1:
                p = mp.Process(name='mapper', target=self.mapping,
                               args=(rank, self.npc, time_string))
            elif rank == 2:
                continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    pass

