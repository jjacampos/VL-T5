#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=mi_fine_tuning

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/fsx/jacampos/experiments/vl-seq2seq/exploration/mi_pretraining/log.out

## filename for job standard error output (stderr)
#SBATCH --error=/fsx/jacampos/experiments/vl-seq2seq/exploration/mi_pretraining/log.err

## partition name
#SBATCH --partition=hipri

## number of gpus
#SBATCH --gpus-per-node=1

#SBATCH --account all

## number of tasks per node
#SBATCH --ntasks-per-node=1


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

# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/fsx/jacampos/experiments/vl-seq2seq/transformers

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=60909 \
        ../src/comet.py \
        --distributed --multiGPU \
        --train_path /fsx/jacampos/data/comet/split_v2/just_mm/mem_dials_gpt2_train.json \
        --valid_path  /fsx/jacampos/data/comet/split_v2/just_mm/mem_dials_gpt2_val.json \
        --test_path /fsx/jacampos/data/comet/split_v2/just_mm/mem_dials_gpt2_test.json\
	    --coco_annotations_path /data/datasets01/COCO/060817/annotations/instances_train2014.json \
	    --memory_files /fsx/jacampos/data/comet/split_v2/memory_may21_v1_100graphs.json /fsx/jacampos/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	    --coco_features_path /fsx/jacampos/data/pretraining/datasets/COCO/features/train2014_obj36.h5 \
	    --special_tokens_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_special_tokens.json \
	    --do_train \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 1 \
        --backbone 't5-base' \
        --num_beams 5 \
        --n_images 5 \
        --batch_size 32 \
        --valid_batch_size 32 \
	--n_boxes 10 \
        --output /fsx/jacampos/experiments/vl-seq2seq/exploration/mi_pretraining/ \
        --load  /fsx/jacampos/experiments/pretraining/warm_start/Epoch31 \
        --randomization no_random \
        --use_mem_ids \
        --match_text_image \
        --run_name mi_pre_training_just_mm \