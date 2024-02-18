#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -p shared
#SBATCH --time=2-00:00:00
#SBATCH --output=/nobackup/wclv88/geography/outs/slurm-%h.out

csv_files=(*.csv)

head -1 "${csv_files[0]}" > output.csv

for csv_file in "${csv_files[@]}"; do
    tail -n +2 "$csv_file" >> output.csv
done