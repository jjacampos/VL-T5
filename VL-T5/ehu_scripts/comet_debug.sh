#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --job-name=vl-t5
#SBATCH --output=debug.out


# The name of experiment
source /gscratch4/users/jcampos004/summer_internship/venv/bin/activate

# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/models/transformers
export MASTER_ADDR=12345

python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    ../src/comet.py \
        --distributed --multiGPU \
        --train_path /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_train.json \
        --valid_path  /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_val.json \
        --test_path /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_test.json\
	--coco_annotations_path /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/coco/annotations/instances_train2014.json \
	--memory_files /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/memory_may21_v1_100graphs.json /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	--coco_features_path /gscratch4/users/jcampos004/summer_internship/features/COCO/features/train2014_obj36.h5 \
	--special_tokens_path  /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_special_tokens.json \
	--do_train \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output /gscratch4/users/jcampos004/summer_internship/models/non_para_vlt5 \
        --load  /gscratch4/users/jcampos004/summer_internship/models/transformers/Epoch30 \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 32 \
	--n_boxes 10 \
        --use_mem_ids \
        --match_text_image \
        --run_name non_paraphrased_vlt5

python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    ../src/comet.py \
        --distributed --multiGPU \
        --train_path /gscratch4/users/jcampos004/summer_internship/data/comet/last_paraphrased/mem_dials_gpt2_train.json \
        --valid_path  /gscratch4/users/jcampos004/summer_internship/data/comet/last_paraphrased/mem_dials_gpt2_val.json \
        --test_path /gscratch4/users/jcampos004/summer_internship/data/comet/last_paraphrased/mem_dials_gpt2_test.json\
	--coco_annotations_path /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/coco/annotations/instances_train2014.json \
	--memory_files /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/memory_may21_v1_100graphs.json /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	--coco_features_path /gscratch4/users/jcampos004/summer_internship/features/COCO/features/train2014_obj36.h5 \
	--special_tokens_path  /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_special_tokens.json \
	--do_train \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output /gscratch4/users/jcampos004/summer_internship/models/para_vlt5 \
        --load  /gscratch4/users/jcampos004/summer_internship/models/transformers/Epoch30 \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 32 \
	--n_boxes 10 \
        --use_mem_ids \
        --match_text_image \
        --run_name paraphrased_vlt5

python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    ../src/comet.py \
        --distributed --multiGPU \
        --train_path /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_train.json \
        --valid_path  /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_val.json \
        --test_path /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_test.json\
	--coco_annotations_path /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/coco/annotations/instances_train2014.json \
	--memory_files /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/memory_may21_v1_100graphs.json /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	--coco_features_path /gscratch4/users/jcampos004/summer_internship/features/COCO/features/train2014_obj36.h5 \
	--special_tokens_path  /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_special_tokens.json \
	--do_train \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output /gscratch4/users/jcampos004/summer_internship/models/non_para_ours \
        --load  /gscratch4/users/jcampos004/summer_internship/models/warm_start_8_images/Epoch37 \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 32 \
	--n_boxes 10 \
        --use_mem_ids \
        --match_text_image \
        --run_name non_parphrased_ours

python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    ../src/comet.py \
        --distributed --multiGPU \
        --train_path /gscratch4/users/jcampos004/summer_internship/data/comet/last_paraphrased/mem_dials_gpt2_train.json \
        --valid_path  /gscratch4/users/jcampos004/summer_internship/data/comet/last_paraphrased/mem_dials_gpt2_val.json \
        --test_path /gscratch4/users/jcampos004/summer_internship/data/comet/last_paraphrased/mem_dials_gpt2_test.json\
	--coco_annotations_path /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/coco/annotations/instances_train2014.json \
	--memory_files /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/memory_may21_v1_100graphs.json /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	--coco_features_path /gscratch4/users/jcampos004/summer_internship/features/COCO/features/train2014_obj36.h5 \
	--special_tokens_path  /sc01a4/users/jcampos004/PERSONAL_STORAGE/summer_internship/data/comet/split_v2/mem_dials_gpt2_special_tokens.json \
	--do_train \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output /gscratch4/users/jcampos004/summer_internship/models/para_ours \
        --load  /gscratch4/users/jcampos004/summer_internship/models/warm_start_8_images/Epoch37 \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 32 \
	--n_boxes 10 \
        --use_mem_ids \
        --match_text_image \
        --run_name parphrased_ours