#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=16GB 
#SBATCH --time=30:00:00 
#SBATCH --partition=open 
#SBATCH --account=cdm8 
#SBATCH --partition=sla-prio

module load anaconda
conda activate /storage/group/cdm8/default/vikas/conda_env/pdes

python dG_moiety_gen_rad1_cmd_run.py


