#!/bin/bash
#SBATCH -n 16                              # Number of cores
#SBATCH --time=80:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=1
#SBATCH --job-name=point-slam-euler-test
#SBATCH --output=/cluster/scratch/liuqing/point-slam/euler.out
#SBATCH --error=/cluster/scratch/liuqing/point-slam/euler.err

JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 

source /cluster/home/liuqing/miniconda3/etc/profile.d/conda.sh

conda activate point-slam

datasets=("ScanNet")

scannet_scenes=("scene0025")

output_affix="/cluster/scratch/liuqing/3DVoutput"
# run.py arguments: [scene config] --gt_camera --wandb --dynamic_r --fixed_r --radius_add_max --radius_add --radius_query
#                   --use_viewdir --no_viewdir --encode_viewdir --no_encode_viewdir --use_color_track --track_w_color_loss 
#                   --use_BA --track_iter --map_iter --map_every --output --map_win_size --kf_every
#                   --kf_selection --eval_img --no_eval_img --project_name --use_exposure --no_exposure --end_correct --no_end_correct
#                   --track_color --track_uniform --rel_pos_in_col --no_rel_pos_in_col --depth_limit --no_depth_limit --min_iter_ratio  

method="point-slam"
dataset=${datasets[2]}
#scene_name=${replica_scenes[0]}
scene_name="scene0181"

run_args="--wandb --project_name POINT_SLAM_hierarchical --map_iter 600 --kf_every 10 --map_win_size 20"

# Run single or array job

#python run.py configs/${dataset}/${scene_name}.yaml $run_args --output ${output_affix}/${method}/${dataset}/${scene_name}-${run_suffix}

python run.py configs/${dataset}/${scene_name}.yaml $run_args --output ${output_affix}/${method}/${dataset}/${scene_name}-${run_suffix}

# python run.py configs/ScanNet/scene0025.yaml --wandb --project_name POINT_SLAM_memory_test --map_iter 600 --kf_every 20 --map_win_size 12 --output /cluster/scratch/liuqing/point-slam-memory/point-slam/ScanNet/scene0025
# Send some noteworthy information to the output log
echo ""
echo "Job Comment:     test NICER-SLAM"
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     ${JOB_START_TIME}"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
