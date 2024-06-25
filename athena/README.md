source ~/athena_env/bin/activate
pip install -r requirements.txt

module list
module avail
module load Python/3.10.4
module unload Python/3.10.4

hpc-grants
hpc-fs
hpc-jobs
hpc-jobs-history

chmod +x image_colorizer.py image_colorizer.sh
sbatch image_colorizer.sh
squeue -u plgtomekszkaradek
squeue

tail -f -n50 logs/log_file.log
