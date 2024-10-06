#!/bin/bash 

# Function to display usage information
usage() {
    echo "Usage: $0 [--neus_only] [--sn_est] CASE_NAME1 [CASE_NAME2 ...]"
    echo "  --neus_only  Only run the second command (exp_runner.py)"
    exit 1
}

# Parse optional --neus_only flag
NEUS_ONLY=false
SN_EST=false
if [ "$1" == "--neus_only" ]; then
    NEUS_ONLY=true
    shift # Remove the flag from the arguments
fi

if [ "$1" == "--sn_est" ]; then
    SN_EST=true
    shift # Remove the flag from the arguments
fi

# Check if at least one CASE_NAME is provided
if [ "$#" -lt 1 ]; then
    usage
fi

# Loop over each provided CASE_NAME
for CASE_NAME in "$@"
do
    echo "Processing CASE_NAME: $CASE_NAME"

    if [ "$NEUS_ONLY" = false ]; then
        # Run the first command
        accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py \
            --config configs/mvdiffusion-joint-ortho-6views.yaml \
            validation_dataset.root_dir=./example_images \
            validation_dataset.filepaths=["$CASE_NAME.png"] \
            save_dir=./outputs
    fi

    # Switch directory to NeuS/
    echo "Switching directory to NeuS/"
    cd NeuS/

    DATASET_NAME="${CASE_NAME%_*}"  # Everything before the last underscore (e.g., table_1_far)
    OBJ_NUMBER="${CASE_NAME##*_}" # The part after the last underscore (e.g., 1)

    if [ "$SN_EST" = true ]; then
        # run surface normal estimation
        # first check if the directory exists
        if [ ! -d "../instant-nsr-pl/datasets/fixed_poses_$DATASET_NAME" ]; then
            echo "Directory '../instant-nsr-pl/datasets/fixed_poses_$DATASET_NAME' does not exist. Running surface normal detection"
            
            # Run test.py with $CASE_NAME as an argument (or modify as needed)
            python estimation_surface_normal.py "$CASE_NAME"
        else
            echo "Using pre-computed fixed camera poses at '../instant-nsr-pl/datasets/fixed_poses_$DATASET_NAME'."
        fi
    fi

    # Run the second command
    python exp_runner.py --mode train --conf ./confs/wmask_ar.conf \
        --case "$CASE_NAME" --data_dir ../outputs/cropsize-192-cfg1.0/

    # Return to the original directory after running the command
    cd - > /dev/null
done
echo "All cases processed successfully."