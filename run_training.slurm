#!/bin/bash
#SBATCH --output /home/roberto/Output.out
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
training_file="training/training.csv"
config_file="config.json"

# Run the Python script
srun python -u training.py $training_file --columns_to_exclude tableName key group --target_column target > Run.log 2>&1

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""
