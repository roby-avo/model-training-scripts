#!/bin/bash
#SBATCH --output /home/Roberto/Output.out
#SBATCH --job-name train
#SBATCH --partition sintef
#SBATCH --ntasks 1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:a30:1
#SBATCH --time 00-24:00:00

# ENABLE ACCESS TO CONDA ENVIRONMENTS
. "/opt/miniconda3/etc/profile.d/conda.sh"

# ACTIVATE CONDA ENVIRONMENT
conda activate neuralnet-env

echo ""
echo "***** LAUNCHING *****"
echo `date '+%F %H:%M:%S'`
echo ""

echo "hostname="`hostname`

echo ""
echo "***"
echo ""

# Define the input arguments
predictions_dir="/path/to/predictions"
model_path="/path/to/model_training.h5"
annotations_file="/path/to/annotations.json"

# Run the Python script
srun python -u compute_performance.py $predictions_dir $model_path $annotations_file > Run.log 2>&1

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""