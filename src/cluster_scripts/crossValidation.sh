#!/bin/sh
#SBATCH --job-name="R18C1F08"
#SBATCH --account=ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="P100"
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=10:00:00
#SBATCH --output=ResNet18_ctype1_f08.log
#SBATCH --mail-user=<marcgro@stud.ntnu.no>
#SBATCH --mail-type=END

module purge
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1

python3 ./WasteClassification/src/crossValidation.py