#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=multi_image_pretraining

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/fsx/jacampos/experiments/pretraining/warm_start/log.out

## filename for job standard error output (stderr)
#SBATCH --error=/fsx/jacampos/experiments/pretraining/warm_start/log.err

## partition name
#SBATCH --partition=hipri

## number of gpus
#SBATCH --gpus-per-node=4

#SBATCH --account all

## number of tasks per node
#SBATCH --ntasks-per-node=8

### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# Start clean
module purge

# Load what we need
source /data/home/jacampos/miniconda/etc/profile.d/conda.sh
conda activate vlt5

# The name of experiment
name=multi_image_pretrain

output=/fsx/jacampos/experiments/pretraining/warm_start/

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    ../src/multi_image_pretrain.py \
        --distributed --multiGPU --fp16 \
        --train mscoco_resplit_train,vgnococo \
        --valid mscoco_resplit_val \
        --batch_size 80 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --losses 'lm,qa,ig,og,cr,ci' \
        --backbone 't5-base' \
        --output $output \
        --epoch 30 \
        --use_mem_ids \
        --match_text_image \
        --n_boxes 10 \
        --load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 


