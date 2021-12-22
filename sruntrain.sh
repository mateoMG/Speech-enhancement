#!/bin/bash
#SBATCH -A plgsethesis
#SBATCH -J train1
#SBATCH --output=/net/archive/groups/plggvoice/unet_logs/output_%j_%a.txt
#SBATCH --error=/net/archive/groups/plggvoice/unet_logs/error_%j_%a.txt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p plgrid-gpu
#SBATCH -t 3-0:00
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=64G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf2

module load plgrid/apps/cuda/10.1
module load plgrid/tools/ffmpeg

export LD_LIBRARY_PATH=/net/archive/groups/plggvoice/cuda/lib64:$LD_LIBRARY_PATH

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

if [ -z ${script+x} ]; then script=trainMG; else echo $script; fi
if [ -z ${cwd+x} ]; then cwd=$(pwd); else echo $cwd; fi

cd $cwd


python $script.py 2>&1 | tee ../unet_logs/unet_log_${timestamp}_$script.txt


