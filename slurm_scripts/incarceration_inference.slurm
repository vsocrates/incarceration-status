#!/bin/bash
#SBATCH --partition=gpu,day
#SBATCH --requeue
#SBATCH --job-name=incarc
#SBATCH --out="%x-%j.out"
#SBATCH --mem-per-cpu=8000
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=5
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --time=00-12:00:00
#SBATCH --mail-type=ALL


# do something
echo "I'm echoing to stdout"
echo "I'm echoing to stderr" 1>&2
echo "My JobID is ${SLURM_JOBID}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s)"

# module restore condacuda101
module load CUDA/10.2.89
module load miniconda
source activate prodigy
cd /home/vs428/Documents/Incarceration 

python incarceration_status_inference.py
