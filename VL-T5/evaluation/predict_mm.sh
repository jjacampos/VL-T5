model_path=$1
predict_path=$2
dataset=$3

source /data/home/jacampos/miniconda/etc/profile.d/conda.sh
conda activate vlt5
# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/fsx/jacampos/experiments/vl-seq2seq/transformers

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port=12346 \
        ../src/comet.py \
        --distributed --multiGPU \
        --train_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_train.json \
        --valid_path  /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_val.json \
        --test_path $dataset\
	--coco_annotations_path /data/datasets01/COCO/060817/annotations/instances_train2014.json \
	--memory_files /fsx/jacampos/data/comet/split_v2/memory_may21_v1_100graphs.json /fsx/jacampos/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	--coco_features_path /fsx/jacampos/data/pretraining/datasets/COCO/features/train2014_obj36.h5 \
	--special_tokens_path /fsx/jacampos/data/comet/paraphrased/original_paraphrased/mem_dials_gpt2_special_tokens.json \
	--do_train \
        --optim adamw \
        --do_test \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 1 \
        --backbone 't5-base' \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 64 \
	--n_boxes 10 \
        --output $predict_path \
        --num_turns 100\
        --load $model_path \
        --randomization no_random \
        --use_mem_ids \
        --match_text_image
