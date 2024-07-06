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
training_file="training_datasets"
training_file_output="training_datasets_output"
config_file="config.json"

# Run the Python script
srun python -u train_on_datasets.py $training_file $training_file_output --columns_to_exclude tableName key group --target_column target --json_config_file $config_file > Run.log 2>&1

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""