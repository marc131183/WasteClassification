#!/bin/sh
#SBATCH --job-name="clusterLearning"
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="P100"
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --time=04:00:00
#SBATCH --output=clusterLearning.log
#SBATCH --mail-user=<marcgro@stud.ntnu.no>
#SBATCH --mail-type=ALL

module purge
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1

python3 ./WasteClassification/src/clusterLearning.py