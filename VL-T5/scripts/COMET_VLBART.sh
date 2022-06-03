# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/fsx/jacampos/experiments/vl-seq2seq/transformers

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/comet.py \
        --distributed --multiGPU \
        --train_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_train.json \
        --valid_path  /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_val.json \
        --test_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_test.json\
	--coco_annotations_path /data/datasets01/COCO/060817/annotations/instances_train2014.json \
	--memory_files /fsx/jacampos/data/comet/split_v2/mem_dials_merged.json mem_dials_test_v2.json \
	--coco_features_path /fsx/jacampos/data/COCO_Features/COCO/features/train2014_obj36.h5 \
	--special_tokens_path /fsx/jacampos/data/comet/split_v2/mem_dials_gpt2_special_tokens.json \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output /fsx/jacampos/experiments/vl-seq2seq/output \
        --load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLBart/Epoch30.pth \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \
