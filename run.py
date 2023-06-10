import argparse
import random
import numpy as np
import torch

from src import config
from src.Point_SLAM import Point_SLAM


def setup_seed(seed):
    torch.manual_seed(seed)                           
    torch.cuda.manual_seed_all(seed)           # Sets the seed for generating random numbers on all GPUs.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    setup_seed(1219)
    parser = argparse.ArgumentParser(
        description='Arguments for running the Point-SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--nice', action='store_true', default=True)
    
    parser.add_argument('--gt_camera', action='store_true')
    parser.add_argument('--fixed_r', action='store_true')
    parser.add_argument('--dynamic_r', action='store_true')
    parser.add_argument('--use_viewdir', action='store_true')
    parser.add_argument('--no_viewdir', action='store_true')
    parser.add_argument('--encode_viewdir', action='store_true')
    parser.add_argument('--no_encode_viewdir', action='store_true')
    parser.add_argument('--use_exposure', action='store_true')
    parser.add_argument('--no_exposure', action='store_true')
    parser.add_argument('--end_correct', action='store_true')
    parser.add_argument('--no_end_correct', action='store_true')
    parser.add_argument('--use_color_track', action='store_true')
    parser.add_argument('--no_color_track', action='store_true')
    parser.add_argument('--use_BA', action='store_true')
    parser.add_argument('--no_BA', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--rel_pos_in_col', action='store_true')
    parser.add_argument('--no_rel_pos_in_col', action='store_true')
    parser.add_argument('--eval_img', action='store_true')
    parser.add_argument('--no_eval_img', action='store_true')
    parser.add_argument('--depth_limit', action='store_true')
    parser.add_argument('--no_depth_limit', action='store_true')
    parser.add_argument('--kf_selection', action='store_true')
    parser.add_argument('--track_color', action='store_true')
    parser.add_argument('--track_uniform', action='store_true')

    # need to use same args in get_mesh_tsdf_fusion
    parser.add_argument('--radius_add_max', type=float) # for dynamic r
    parser.add_argument('--radius_add', type=float) # for fixed r
    parser.add_argument('--radius_query', type=float) # for fixed r
    parser.add_argument('--track_w_color_loss', type=float)
    parser.add_argument('--track_iter', type=int)
    parser.add_argument('--map_iter', type=int)
    parser.add_argument('--min_iter_ratio', type=float)
    parser.add_argument('--map_every', type=int)
    parser.add_argument('--kf_every', type=int)
    parser.add_argument('--map_win_size', type=int)
    parser.add_argument('--kf_t_thre', type=float)
    parser.add_argument('--kf_r_thre', type=float)
    parser.add_argument('--project_name', type=str)

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/point_slam.yaml')

    slam = Point_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
