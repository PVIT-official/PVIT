BASE_PATH=$(cd "$(dirname "$0")"; pwd)
BASE_PATH=${BASE_PATH%%/scripts*}

cd $BASE_PATH

PYTHONPATH=$BASE_PATH torchrun --nnodes=1 --nproc_per_node=8 --master_port=$PORT \
    pvit/train/train_mem.py \
    --model_name_or_path $MODEL_PATH \
    --version v1 \
    --train_data_paths $DATA_PATH \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_region_clip $REGION_CLIP_PATH \
    --mm_use_bbox_fc True \
    --tune_mm_prompt_only False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --ddp_find_unused_parameters True \
    --report_to none
