#!/bin/sh
#SBATCH --job-name="stratifiedKFold"
#SBATCH --partition=CPUQ
#SBATCH --mem=12000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=stratifiedKFold.log
#SBATCH --mail-user=<marcgro@stud.ntnu.no>
#SBATCH --mail-type=ALL

module purge

python3 ./WasteClassification/src/stratifiedKFold.py