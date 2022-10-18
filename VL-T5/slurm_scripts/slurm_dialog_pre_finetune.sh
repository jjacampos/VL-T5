#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=dial_pre

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/fsx/jacampos/experiments/dialog_pretraining/dial_pre_%A_%a.out

## filename for job standard error output (stderr)
#SBATCH --error=/fsx/jacampos/experiments/dialog_pretraining/dial_pre_%A_%a.err

## partition name
#SBATCH --partition=hipri

## number of gpus
#SBATCH --gpus-per-node=4

#SBATCH --account all

## number of tasks per node
#SBATCH --ntasks-per-node=8

#SBATCH --array=1-12

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
base_path="/fsx/jacampos/experiments/dialog_pretraining/"
paths=(\
"coherence,reordering,entities,mm_coherence,mm_reordering" \
"coherence,reordering,mm_coherence,mm_reordering" \
"coherence,speaker,reordering,entities" \
"coherence,reordering,entities" \
"coherence,reordering" \
"reordering" \
"coherence,speaker,reordering,entities,mm_coherence,mm_reordering" \
"coherence,speaker,reordering,entities_class,mm_coherence,mm_reordering" \
"mm_coherence,mm_reordering" \
"coherence,speaker,reordering,entities_OURS" \
"coherence,speaker,reordering,entities_text_T5" \
"coherence,speaker,reordering,entities_T5") 
dialog_losses=(\
"coherence,reordering,entities,mm_coherence,mm_reordering" \
"coherence,reordering,mm_coherence,mm_reordering" \
"coherence,speaker,reordering,entities" \
"coherence,reordering,entities" \
"coherence,reordering" \
"reordering" \
"coherence,speaker,reordering,entities,mm_coherence,mm_reordering" \
"coherence,speaker,reordering,entities_class,mm_coherence,mm_reordering" \
"mm_coherence,mm_reordering" \
"coherence,speaker,reordering,entities" \
"coherence,speaker,reordering,entities" \
"coherence,speaker,reordering,entities") 
hyperparams=(\
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30" \
"--load /fsx/jacampos/experiments/pretraining/warm_start_8_images/Epoch34 --multi_image_pretrain" \
"--just_text_features --just_text_model" \
"--just_text_model")

master_port=(21212 23232 24242 25252 26262 27272 12121 13131 14141 15151 16161 17171 34343)
echo $SLURM_ARRAY_TASK_ID
echo ${paths[$SLURM_ARRAY_TASK_ID-1]}
mkdir -p $base_path${paths[$SLURM_ARRAY_TASK_ID-1]}
mkdir -p $base_path${paths[$SLURM_ARRAY_TASK_ID-1]}
echo $base_path${paths[$SLURM_ARRAY_TASK_ID-1]}

# The name of experiment

# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/fsx/jacampos/experiments/vl-seq2seq/transformers

python -m torch.distributed.launch \
         --nproc_per_node=4 \
        --master_port=${master_port[$SLURM_ARRAY_TASK_ID-1]} \
        ../src/dialog_pre_training.py \
        --distributed --multiGPU \
        --train_path /fsx/jacampos/data/comet/dialog_pre/mem_dials_train.json \
        --valid_path  /fsx/jacampos/data/comet/dialog_pre/mem_dials_val.json \
        --test_path /fsx/jacampos/data/comet/dialog_pre/mem_dials_test.json \
	--coco_annotations_path /data/datasets01/COCO/060817/annotations/instances_train2014.json \
	--memory_files /fsx/jacampos/data/comet/split_v2/memory_may21_v1_100graphs.json /fsx/jacampos/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	--coco_features_path /fsx/jacampos/data/pretraining/datasets/COCO/features/train2014_obj36.h5 \
	--special_tokens_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_special_tokens.json \
	--do_train \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 40 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output $base_path${paths[$SLURM_ARRAY_TASK_ID-1]} \
        --num_beams 5 \
        --batch_size 16 \
        --valid_batch_size 60 \
	--n_boxes 10 \
        --use_mem_ids \
        --match_text_image \
        --run_name ${paths[$SLURM_ARRAY_TASK_ID-1]} \
        --dialog_losses ${dialog_losses[$SLURM_ARRAY_TASK_ID-1]} \
        ${hyperparams[$SLURM_ARRAY_TASK_ID-1]}