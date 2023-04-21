#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=incarc_cancer
#SBATCH --out="moorecancer_%x-%j.out"
#SBATCH --mem=20G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=00-00:10:00
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
cd /home/vs428/Documents/Moore 

# export PRODIGY_HOME="/home/vs428/Documents/Moore"

# python -m spacy train config.cfg --output /home/vs428/project/Moore_models/moore_followup_textcat_v2 --paths.train ./train.spacy --paths.dev ./dev.spacy

python -m prodigy train incarceration_model --ner incarceration_status --label-stats --base-model en_core_web_trf --gpu-id 0 --verbose #  --config /home/vs428/project/Moore_models/task_4_textcat_GOLD/model-best/config.cfg
# ~/Documents/n2c2_2022/annotations/n2c2_planSubsection_train_merged_v2.jsonl
# python -m prodigy train assessment_model --ner n2c2_assessment --label-stats --base-model en_core_web_trf --gpu-id 0
