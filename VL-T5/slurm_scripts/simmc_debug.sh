# The name of experiment

# Run this script by passing the number of processes as first argument
export TRANSFORMERS_CACHE=/fsx/jacampos/experiments/vl-seq2seq/transformers
export MASTER_ADDR=12345

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    ../src/simcc.py \
        --distributed --multiGPU \
        --train_path /fsx/jacampos/data/simmc2/data/simmc_dials_gpt2_train.json \
        --valid_path  /fsx/jacampos/data/simmc2/data/sim_dials_gpt2_dev.json \
        --test_path /fsx/jacampos/data/simmc2/data/sim_dials_gpt2_test_for_challenge.json\
	--features_path /fsx/jacampos/data/simmc2/data/simmc_features.h5py \
	--special_tokens_path /fsx/jacampos/data/simmc2/data/simmc_dials_gpt2_special_tokens.json \
	--do_train \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output /fsx/jacampos/experiments/simmc \
        --load  /fsx/jacampos/experiments/vl-seq2seq/pretrain/snap/pretrain/VLT5/Epoch30 \
        --num_beams 5 \
        --batch_size 30 \
        --valid_batch_size 60 \
	--n_boxes 10 \
        --use_mem_ids \
        --match_text_image 