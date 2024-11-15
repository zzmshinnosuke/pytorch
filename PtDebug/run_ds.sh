deepspeed --num_nodes=1 --num_gpus=1 \
    deepspeed_debug.py \
    --deepspeed "./config/default_offlload_zero2.json" \
    --model_name_or_path MyTsbir \
    --output_dir hhh