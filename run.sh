#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=6 #
#SBATCH --mem=128G 
#SBATCH -o /network/scratch/a/anita.kriz/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load python/3.10

# 2. Activate the venv10 environment
source /network/scratch/a/anita.kriz/venvs/vccrl_env/bin/activate

# 3. Run your script with the same command you'd use interactively
python main.py --config medmcqa.yml --log_folder test --trial_name test_1 --train --ni 

# 4. Evaluate
# python main.py --config medmcqa.yml --log_folder test --trial_name test_1 --test --ni 