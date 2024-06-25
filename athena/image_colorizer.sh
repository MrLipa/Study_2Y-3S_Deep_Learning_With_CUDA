#!/bin/bash

#SBATCH --job-name=image_colorizer
#SBATCH --output=output/result_%j.txt
#SBATCH --error=output/error_%j.txt
#SBATCH --time=00:20:00
#SBATCH --partition=plgrid-now
#SBATCH --gres=gpu

module load libglvnd/1.4.0 libGLU/9.0.2

source ~/athena_env/bin/activate

srun python image_colorizer.py

deactivate

module unload libglvnd/1.4.0 libGLU/9.0.2
