#!/bin/sh
#SBATCH --job-name="clusterLearning"
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=clusterLearning.log
#SBATCH --mail-user=<marcgro@stud.ntnu.no>
#SBATCH --mail-type=ALL

module purge
module load PyTorch/1.7.1-fosscuda-2020b
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1

python3 ./WasteClassification/src/clusterLearning.py