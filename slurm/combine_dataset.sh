#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -p shared
#SBATCH --time=2-00:00:00
#SBATCH --output=/nobackup/wclv88/geography/outs/slurm-%h.out

OUTPUT_DIR=/nobackup/wclv88/geography/datasets/dataset_${SLURM_ARRAY_TASK_ID}

# Get a list of all the subdirectories in OUTPUT_DIR
subdirs=$(find $OUTPUT_DIR -type d)

# Get a list of all the unique filenames in those directories
filenames=$(find $OUTPUT_DIR -type f -printf "%f\n" | sort -u)

# For each unique filename
for filename in $filenames; do
    # Initialize a variable to keep track of whether we've written the header yet
    header_written=false

    # For each subdirectory
    for subdir in $subdirs; do
        # If the file exists in that directory
        if [ -f "$subdir/$filename" ]; then
            # If we haven't written the header yet
            if ! $header_written; then
                # Write the entire file (including the header) to the output file
                cat "$subdir/$filename" > "$OUTPUT_DIR/$filename"
                # And mark the header as written
                header_written=true
            else
                # Otherwise, write the file without the header to the output file
                tail -n +2 "$subdir/$filename" >> "$OUTPUT_DIR/$filename"
            fi
        fi
    done
done