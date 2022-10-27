#!/bin/sh
#SBATCH --job-name="stratifiedKFold"
#SBATCH --partition=CPUQ
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=00:15:00
#SBATCH --output=stratifiedKFold.log
#SBATCH --mail-user=<marcgro@stud.ntnu.no>
#SBATCH --mail-type=ALL

module purge

python3 ./WasteClassification/src/stratifiedKFold.py