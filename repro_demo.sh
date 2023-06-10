#!/bin/bash
#SBATCH -n 16                              # Number of cores
#SBATCH --time=80:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=1
#SBATCH --job-name=point-slam-euler-demo
#SBATCH --output=/cluster/scratch/liuqing/point-slam/euler.out  # to be modified
#SBATCH --error=/cluster/scratch/liuqing/point-slam/euler.err   # to be modified

JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 

source /cluster/home/liuqing/miniconda3/etc/profile.d/conda.sh  # to be modified
conda activate point-slam   # to be modified
datasets=("ScanNet")
output_affix="/cluster/scratch/liuqing/3DVoutput" # to be modified 

method="hierarchical-point-slam"
dataset=${datasets[0]}
scene_name="scene0181"
run_args="--wandb --project_name Hierarchical_Point_SLAM"
python run.py configs/${dataset}/${scene_name}.yaml $run_args --output ${output_affix}/${method}/${dataset}/${scene_name}-${run_suffix}
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
