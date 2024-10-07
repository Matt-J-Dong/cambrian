#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=4                # uses 4 compute cores per task
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=cambrian_gqa
#SBATCH --output=cambrian_gqa.out
#SBATCH --error=cambrian_gqa-error.txt
#SBATCH --mail-user=matthew.dong@nyu.edu
#SBATCH --mail-type=FAIL

singularity exec --nv --overlay $SCRATCH/new_env/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
cd ..
source venv/bin/activate
cd ./eval

# Verify environment
pwd
env_list=\$(conda env list)
echo "\$env_list"

# Check GPU status
nvidia-smi

# Source .env if it exists
if [ -f .env ]; then
    source .env
fi

# Run your script
bash scripts/run_benchmark.sh --benchmark gqa --ckpt nyu-visionx/cambrian-8b --conv_mode llama_3
"