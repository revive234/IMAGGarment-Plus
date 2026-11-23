export HOST_NUM=1
accelerate launch --gpu_ids 0,1,2,3 --use_deepspeed --num_processes 4 --main_process_port 12016 \
  --deepspeed_config_file "[your deepspeed_config_file]" \
  train.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_json_path="[your dataset_json_path]" \
  --image_root_path="[your image_root_path]" \
  --clip_penultimate=False \
  --train_batch_size=10 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000000 \
  --learning_rate=1e-5 \
  --weight_decay=0.01 \
  --lr_scheduler="constant" --num_warmup_steps=2000 \
  --output_dir="[your output_dir]" \
  --checkpointing_steps=1000 \
  --width=512 \
  --height=640
# To resume, append: --resume_from_checkpoint="[your checkpoint_path]"

