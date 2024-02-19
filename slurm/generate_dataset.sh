#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -p shared
#SBATCH --time=2-00:00:00
#SBATCH --output=/nobackup/wclv88/geography/outs/slurm-%A_%a.out
#SBATCH --array=0-99

module load python/3.10.8
module load $PYTHON_BUILD_MODULES

COUNT=50

stdbuf -oL \
python3 \
geography/generate_dataset.py \
    --output_file /nobackup/wclv88/geography/datasets/dataset_${SLURM_ARRAY_TASK_ID}.csv \
    --skip $((SLURM_ARRAY_TASK_ID * COUNT)) \
    --batch $COUNT \
    --city_count 1000 \
    --get_middle_city \ 
    --no_mp