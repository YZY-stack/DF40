export MODEL_NAME="/apdcephfs_cq10/model_weights/stable-diffusion-2-1"

accelerate launch --mixed_precision="fp16" train_scripts/train_sd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="/apdcephfs_cq10/lora_sd2.1_CelebDF" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=16 \
  --max_train_steps=20000 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --validation_prompt="a human face" \
  --checkpointing_steps=2500 \
  --image_folder="/apdcephfs_cq10/Datasets/Celeb-DF-v2/all_frames/" \
  --validation_epochs=1000000


accelerate launch --mixed_precision="fp16" train_scripts/train_sd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="/apdcephfs_cq10/lora_sd2.1_FF" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=16 \
  --max_train_steps=20000 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --validation_prompt="a human face" \
  --checkpointing_steps=2500 \
  --image_folder="/apdcephfs_cq10/Datasets/FaceForensics++/all_frames/" \
  --validation_epochs=1000000
