# The name of experiment
name=VLT5

output=/fsx/jacampos/experiments/pretraining

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    ../src/multi_image_pretrain.py \
        --distributed --multiGPU --fp16 \
        --train mscoco_resplit_train,vgnococo \
        --valid mscoco_resplit_val \
        --batch_size 320 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --losses 'lm,qa,ig,og,cr,ci' \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --epoch 30 \
        --use_mem_ids \
        --match_text_image \
        --n_boxes 10 \
        --load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 \
        --individual_vis_layer_norm true 
