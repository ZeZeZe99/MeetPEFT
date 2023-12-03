torchrun --nproc_per_node=1 supervised-fine-tune-qlora.py  \
        --model_name_or_path daryl149/llama-2-7b-chat-hf \
        --bf16 True \
        --output_dir ./checkpoints/llama2-16k       \
        --model_max_length 16384 \
        --use_flash_attn True \
        --data_path ./data/mb/train_text_lora.json \
        --low_rank_training True \
        --num_train_epochs 5  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 1     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 200     \
        --save_total_limit 1     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --report_to "wandb" \
        --logging_steps 5     \
        --deepspeed "ds_configs/stage2.json" \
        --tf32 True \
        --cache_dir ./cache