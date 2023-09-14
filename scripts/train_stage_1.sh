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
    --tune_mm_prompt_only True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True  \
    --report_to none
