import sys
sys.path.append('.')
from src.common import as_intrinsics_matrix
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.Point_SLAM import Point_SLAM
from src import config
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import torch
import numpy as np
import argparse
import random
import os
import subprocess
import traceback

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_neural_point_cloud(slam, ckpt, device):

    slam.npc._cloud_pos = ckpt['cloud_pos']
    slam.npc._cloud_normal = ckpt['cloud_normal']
    slam.npc._input_pos = ckpt['input_pos']
    slam.npc._input_rgb = ckpt['input_rgb']
    slam.npc._input_normal = ckpt['input_normal']
    slam.npc._input_normal_cartesian = ckpt['input_normal_cartesian']
    slam.npc._pts_num = len(ckpt['cloud_pos'])
    slam.npc.geo_feats = ckpt['geo_feats'].to(device)
    slam.npc.col_feats = ckpt['col_feats'].to(device)

    cloud_pos = torch.tensor(ckpt['cloud_pos'], device=device)
    slam.npc.index_train(cloud_pos)
    slam.npc.index.add(cloud_pos)

    print(
        f'Successfully loaded neural point cloud, {slam.npc.index.ntotal} points in total.')


def load_ckpt(cfg, slam):
    """
    Saves mesh of already reconstructed model from checkpoint file. Makes it 
    possible to remesh reconstructions with different settings and to draw the cameras
    """
    assert cfg['mapping']['save_selected_keyframes_info'], 'Please save keyframes info to help run this code.'

    ckptsdir = f'{slam.output}/ckpts'
    device = cfg['mapping']['device']
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('\nGet ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
        else:
            raise ValueError(f'Check point directory {ckptsdir} is empty.')
    else:
        raise ValueError(f'Check point directory {ckptsdir} not found.')

    return ckpt


class DepthImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('depth_')])
        self.image_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('color_')])

        indices = []
        for depth_file in self.depth_files:
            base, ext = os.path.splitext(depth_file)
            index = int(base[-5:])
            indices.append(index)
        self.indices = indices

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth = np.load(self.depth_files[idx])
        image = np.load(self.image_files[idx])

        if self.transform:
            depth = self.transform(depth)
            image = self.transform(image)

        return depth, image


def main():
    parser = argparse.ArgumentParser(
        description="Configs for Point-SLAM."
    )
    parser.add_argument(
        "config", type=str, help="Path to config file.",
    )
    parser.add_argument("--input_folder", type=str,
                        help="input folder, this have higher priority, can overwrite the one in config file.",
                        )
    parser.add_argument("--output", type=str,
                        help="output folder, this have higher priority, can overwrite the one in config file.",
                        )
    parser.add_argument("--name", type=str,
                        help="specify the name of the mesh",
                        )
    parser.add_argument("--no_render", default=False, action='store_true',
                        help="if to render frames from checkpoint for constructing the mesh.",
                        )
    parser.add_argument("-s", "--silent", default=False, action='store_true',
                        help="if to print status message.",
                        )
    parser.add_argument("--no_eval", default=False, action='store_true',
                        help="if to evaluate the mesh by 2d and 3d metrics.",
                        )
    parser.add_argument("--mid_mesh", default=False, action='store_true',
                        help="if to extract intermediate mesh.",
                        )

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

    parser.add_argument('--radius_add_max', type=float)
    parser.add_argument('--radius_add', type=float)
    parser.add_argument('--radius_query', type=float)
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
    assert torch.cuda.is_available(), 'GPU required for reconstruction.'
    cfg = config.load_config(args.config, "configs/point_slam.yaml")
    device = cfg['mapping']['device']

    slam = Point_SLAM(cfg, args, share_npc=False,
                      share_decoders=False if args.no_render else True)
    slam.output = cfg['data']['output'] if args.output is None else args.output
    ckpt = load_ckpt(cfg, slam)

    render_frame = not args.no_render
    if render_frame:
        load_neural_point_cloud(slam, ckpt, device)
        idx = 0
        frame_cnt = 0
        K = as_intrinsics_matrix([fx, fy, cx, cy])

        try:
            slam.shared_decoders.load_state_dict(ckpt['decoder_state_dict'])
            if not args.silent:
                print('Successfully loaded decoders.')
        except Exception as e:
            print(e)
        frame_reader = get_dataset(cfg, args, cfg['scale'], device=device)
        visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                vis_dir=os.path.join(slam.output, 'rendered_every_frame'), renderer=slam.renderer_map,
                                verbose=slam.verbose, device=device, wandb=False)

        if not args.silent:
            print('Starting to render frames...')
        last_idx = (ckpt['idx']+1) if (ckpt['idx'] +
                                       1) < len(frame_reader) else len(frame_reader)
        while idx < last_idx:
            _, gt_color, gt_depth, gt_c2w = frame_reader[idx]
            cur_c2w = ckpt['estimate_c2w_list'][idx].to(device)
            cloud_normals = torch.tensor(
                slam.npc.cloud_normal(), device=device)

            cur_frame_depth, cur_frame_color = visualizer.vis_value_only(idx, 0, gt_depth, gt_color, cur_c2w, slam.npc, slam.shared_decoders,
                                                                         slam.npc.geo_feats, slam.npc.col_feats, freq_override=True, normals=cloud_normals)
            np.save(f'{slam.output}/rendered_every_frame/depth_{idx:05d}',
                    cur_frame_depth.cpu().numpy())
            np.save(f'{slam.output}/rendered_every_frame/color_{idx:05d}',
                    cur_frame_color.cpu().numpy())
            idx += cfg['mapping']['every_frame']
            frame_cnt += 1
        if not args.silent:
            print(f'Finished rendering {frame_cnt} frames.')

    dataset = DepthImageDataset(root_dir=slam.output+'/rendered_every_frame')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_folder = cfg['data']['input_folder'] if args.input_folder is None else args.input_folder
    scene_name = input_folder.split('/')[-1]
    mesh_name = f'{scene_name}_pred_mesh.ply' if args.name is None else args.name
    mesh_out_file = f'{slam.output}/mesh/{mesh_name}'
    mesh_align_file = f'{slam.output}/mesh/mesh_tsdf_fusion_aligned.ply'

    H, W, fx, fy, cx, cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
    scale = 1.0
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    if not args.silent:
        print('Starting to integrate the mesh...')
    cam_points = []
    # address the misalignment in open3d marching cubes
    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                         scale / 512.0, -2.5 * scale / 512.0)
    if args.mid_mesh:
        os.makedirs(f'{slam.output}/mesh/mid_mesh', exist_ok=True)
    for i, (depth, color) in enumerate(dataloader):
        index = dataset.indices[i]
        depth = depth[0].cpu().numpy()
        color = color[0].cpu().numpy()
        c2w = ckpt['estimate_c2w_list'][index].cpu().numpy()

        c2w[:3, 1] *= -1.0
        c2w[:3, 2] *= -1.0
        w2c = np.linalg.inv(c2w)
        cam_points.append(c2w[:3, 3])

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(
            np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=30,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)
        if args.mid_mesh:
            if i > 0 and (i % 4) == 0:
                o3d_mesh = volume.extract_triangle_mesh()
                o3d_mesh = o3d_mesh.translate(compensate_vector)
                o3d.io.write_triangle_mesh(
                    f'{slam.output}/mesh/mid_mesh/frame_{5*i}_mesh.ply', o3d_mesh)
                print(f"saved intermediate mesh until frame {5*i}.")

    o3d_mesh = volume.extract_triangle_mesh()
    np.save(os.path.join(f'{slam.output}/mesh',
            'vertices_pos.npy'), np.asarray(o3d_mesh.vertices))
    o3d_mesh = o3d_mesh.translate(compensate_vector)

    o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
    if not args.silent:
        print('🕹️ Meshing finished.')

    eval_recon = not args.no_eval
    if eval_recon:
        try:
            if cfg['dataset'] == 'replica':
                print('Evaluating...')
                # result_recon_obj = subprocess.run(['python', '-u', 'src/tools/eval_recon.py', '--rec_mesh',
                #                                    mesh_out_file,
                #                                    '--gt_mesh', f'cull_replica_mesh/{scene_name}.ply', '-3d', '-2d'],
                #                                   text=True, check=True, capture_output=True)
                result_recon_obj = subprocess.run(['python', '-u', 'src/tools/eval_recon.py', '--rec_mesh',
                                                   mesh_out_file,
                                                   '--gt_mesh', f'/cluster/project/infk/courses/252-0579-00L/group10_2023/cull_replica_mesh/{scene_name}.ply', '-3d', '-2d'],
                                                  text=True, check=True, capture_output=True)
                
                result_recon = result_recon_obj.stdout
                print(result_recon)
                print('✨ Successfully evaluated 3D reconstruction.')
            else:
                print('Current dataset not supported for evaluating 3D reconstruction.')
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            print('Failed to evaluate 3D reconstruction.')


if __name__ == "__main__":
    main()
