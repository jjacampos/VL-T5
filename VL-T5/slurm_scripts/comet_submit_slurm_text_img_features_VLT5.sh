#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=no_mem_ids_exploration

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/fsx/jacampos/experiments/vl-seq2seq/exploration/no_mem_ids_exploration_%A_%a.out

## filename for job standard error output (stderr)
#SBATCH --error=/fsx/jacampos/experiments/vl-seq2seq/exploration/no_mem_ids_exploration_%A_%a.err

## partition name
#SBATCH --partition=hipri

## number of gpus
#SBATCH --gpus-per-node=1

#SBATCH --account all

## number of tasks per node
#SBATCH --ntasks-per-node=1

#SBATCH --array=1-1

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
base_path="/fsx/jacampos/experiments/vl-seq2seq/exploration/"
paths=("vlt5/text_image_features/no_mem_ids/optimize_ja_full_turns/global" \
"vlt5/text_image_features/no_mem_ids/optimize_ja_full_turns/normal" \
"vlt5/text_image_features/no_img_text_matching/optimize_ja_full_turns/global" \
"vlt5/text_image_features/no_img_text_matching/optimize_ja_full_turns/normal" \
"vlt5/text_image_features/img_text_matching/optimize_ja_full_turns/global" \
"vlt5/text_image_features/img_text_matching/optimize_ja_full_turns/normal" \
"t5/text_image_features/no_mem_ids/optimize_ja_full_turns/global" \
"t5/text_image_features/no_mem_ids/optimize_ja_full_turns/normal" \
"t5/text_image_features/no_img_text_matching/optimize_ja_full_turns/global" \
"t5/text_image_features/no_img_text_matching/optimize_ja_full_turns/normal" \
"t5/text_image_features/img_text_matching/optimize_ja_full_turns/global" \
"t5/text_image_features/img_text_matching/optimize_ja_full_turns/normal" \
"mi_pretraining/text_image_features/img_text_matching/normal/optimize_ja_full_turns/warm" \
"mi_pretraining/text_image_features/img_text_matching/normal/optimize_ja_full_turns/scratch")
hyperparams=("--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --randomization random_global --run_name vlt5_mm_no_mem_global_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --randomization no_random --run_name vlt5_mm_no_mem_normal_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --randomization random_global --use_mem_ids --run_name vlt5_mm_no_matching_global_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --randomization no_random --use_mem_ids --run_name vlt5_mm_no_matching_normal_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --randomization random_global --use_mem_ids --match_text_image --run_name vlt5_mm_matching_global_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --randomization no_random --use_mem_ids --match_text_image --run_name vlt5_mm_matching_normal_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --just_text_model --randomization random_global --run_name t5_mm_no_mem_global_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --just_text_model --randomization no_random --run_name t5_mm_no_mem_normal_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --just_text_model --randomization random_global --use_mem_ids --run_name t5_mm_no_matching_global_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --just_text_model --randomization no_random --use_mem_ids --run_name t5_mm_no_matching_normal_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --just_text_model --randomization random_global --use_mem_ids --match_text_image --run_name t5_mm_matching_global_ja_full" \
"--load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 --just_text_model --randomization no_random --use_mem_ids --match_text_image --run_name t5_mm_matching_normal_ja_full" \
"--load  /fsx/jacampos/experiments/pretraining/warm_start/Epoch58 --randomization no_random --use_mem_ids --match_text_image --run_name mi_pretraining_mm_matching_normal_warm_ja" \
"--load  /fsx/jacampos/experiments/pretraining/scratch/Epoch30 --randomization no_random --use_mem_ids --match_text_image --run_name mi_pretraining_mm_matching_normal_cold_ja")
master_port=(11111 22222 33333 44444 55555 17171 14141 19191 16161 10101 12121 13131 14141 15151)
echo $SLURM_ARRAY_TASK_ID
echo ${paths[$SLURM_ARRAY_TASK_ID-1]}
mkdir -p $base_path${paths[$SLURM_ARRAY_TASK_ID-1]}/valid
mkdir -p $base_path${paths[$SLURM_ARRAY_TASK_ID-1]}/test
echo $base_path${paths[$SLURM_ARRAY_TASK_ID-1]}
# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/fsx/jacampos/experiments/vl-seq2seq/transformers

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=${master_port[$SLURM_ARRAY_TASK_ID-1]} \
        ../src/comet.py \
        --distributed --multiGPU \
        --train_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_train.json \
        --valid_path  /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_val.json \
        --test_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_test.json\
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
        --batch_size 32 \
        --valid_batch_size 32 \
	--n_boxes 10 \
        --output $base_path${paths[$SLURM_ARRAY_TASK_ID-1]} \
        --optimize_ja \
        --num_turns 100 \
        ${hyperparams[$SLURM_ARRAY_TASK_ID-1]}
       

