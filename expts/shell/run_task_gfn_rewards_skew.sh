#!/bin/bash
#SBATCH -o /mnt/ps/home/CORP/lazar.atanackovic/gflownet/expts/slurm_logs_gfn_rewards_skew/log-%A-%a.out
#SBATCH --job-name=gfn
#SBATCH --partition=long
#SBATCH --gpus-per-node=1080ti:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal

# activate conda environment
source venv-gfn/bin/activate

# Launch these jobs with sbatch --array=0-N%M job.sh   (N is inclusive, M limits number of tasks run at once)
srun python expts/task_gfn_rewards_skew.py $SLURM_ARRAY_TASK_ID