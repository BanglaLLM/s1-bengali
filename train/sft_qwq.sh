# Set environment variables for better multi-GPU performance
base_model="Qwen/QwQ-32B"
dataset_name="BanglaLLM/s1k-Bangla_tokenized-qwq32b"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1  # With H200, this should be fine at 1
block_size=16384  # Start with a moderate block size and scale up if successful
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

# Clear GPU cache before starting
nvidia-smi -r

torchrun --nproc-per-node ${gpu_count} --nccl_p2p_disable=0 --nccl_ib_disable=0 --master_port 12345 \
    train/sft.py \
    --block_size=${block_size} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path=${dataset_name} \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/s1-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --optim="paged_adamddw_32bit"
