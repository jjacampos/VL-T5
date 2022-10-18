# The name of experiment

# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/fsx/jacampos/experiments/vl-seq2seq/transformers
export MASTER_ADDR=12345

python -m torch.distributed.launch \
    --nproc_per_node=1 \
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
        --epochs 10 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output /fsx/jacampos/experiments/dialog_pretraining \
        --load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 \
        --num_beams 5 \
        --batch_size 20 \
        --valid_batch_size 60 \
	--n_boxes 10 \
        --use_mem_ids \
        --match_text_image \
        --dialog_losses 'coherence,speaker,reordering,entities,mm_coherence,mm_reordering' \