#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=reparaphrased_exploration

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/fsx/jacampos/experiments/reparaphrased/reparaphrase_%A_%a.out

## filename for job standard error output (stderr)
#SBATCH --error=/fsx/jacampos/experiments/reparaphrased/reparaphrase_%A_%a.err

## partition name
#SBATCH --partition=hipri

## number of gpus
#SBATCH --gpus-per-node=1

#SBATCH --account all

## number of tasks per node
#SBATCH --ntasks-per-node=1

#SBATCH --array=1-4

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
base_path="/fsx/jacampos/experiments/reparaphrased/our_pretraining"
paths=(\
"scratch_4_context" \
"warm_start_4_context" \
"warm_start_14_context" \
"warm_start_8_context" )
run_names=(\
"scratch_4_context" \
"warm_start_4_context" \
"warm_start_14_context" \
"warm_start_8_context" )
hyperparams=(\
"--load /fsx/jacampos/experiments/pretraining/scratch/Epoch30 " \
"--load /fsx/jacampos/experiments/pretraining/warm_start/Epoch58" \
"--load /fsx/jacampos/experiments/pretraining/warm_start_15_images/Epoch65" \
"--load /fsx/jacampos/experiments/pretraining/warm_start/Epoch35" )
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
        --train_path /fsx/jacampos/data/comet/paraphrased/original_paraphrased/mem_dials_gpt2_train.json \
        --valid_path  /fsx/jacampos/data/comet/paraphrased/original_paraphrased/mem_dials_gpt2_val.json \
        --test_path /fsx/jacampos/data/comet/paraphrased/original_paraphrased/mem_dials_gpt2_test.json \
        --coco_annotations_path /data/datasets01/COCO/060817/annotations/instances_train2014.json \
	--memory_files /fsx/jacampos/data/comet/split_v2/memory_may21_v1_100graphs.json /fsx/jacampos/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	--coco_features_path /fsx/jacampos/data/pretraining/datasets/COCO/features/train2014_obj36.h5 \
	--special_tokens_path /fsx/jacampos/data/comet/paraphrased/original_paraphrased/mem_dials_gpt2_special_tokens.json \
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
        --num_turns 100 \
        --use_mem_ids \
        --match_text_image \
        --multi_image_pretrain \
        --run_name ${run_names[$SLURM_ARRAY_TASK_ID-1]} \
        ${hyperparams[$SLURM_ARRAY_TASK_ID-1]}
       

