#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=incarc
#SBATCH --out="%x-%j.out"
#SBATCH --mem=75G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=00-06:00:00
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

python -m spacy train /home/vs428/Documents/Incarceration/incarceration_status_binary_v2/config.cfg --output incarceration_model_binary_trf_v2 --verbose --gpu-id 0