# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/fsx/jacampos/experiments/vl-seq2seq/transformers
python -m torch.distributed.launch \
        --nproc_per_node=$1 \
        --master_port=12349 \
        ../src/comet.py \
        --distributed --multiGPU \
        --train_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_train.json \
        --valid_path  /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_val.json \
        --test_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_test.json\
	--coco_annotations_path /data/datasets01/COCO/060817/annotations/instances_train2014.json \
	--memory_files /fsx/jacampos/data/comet/split_v2/memory_may21_v1_100graphs.json /fsx/jacampos/data/comet/split_v2/mscoco_memory_graphs_1k.json\
	--coco_features_path /fsx/jacampos/data/COCO_Features/COCO/features/train2014_obj36.h5 \
	--special_tokens_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_special_tokens.json \
	--do_train \
	--optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 10 \
        --num_workers 1 \
        --backbone 'facebook/bart-base' \
        --output /fsx/jacampos/experiments/vl-seq2seq/output_bart_36_features \
        --load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLBart/Epoch30 \
        --num_beams 5 \
        --batch_size 40 \
        --valid_batch_size 100 \
        --use_mem_ids \
