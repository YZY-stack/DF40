export MODEL_NAME="/apdcephfs_cq8/ckpt/stable-diffusion-xl-base-1.0"
export VAE_NAME="/apdcephfs_cq8/ckpt/sdxl-vae-fp16-fix"

accelerate launch --mixed_precision="fp16" train_scripts/train_sdxl_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --output_dir="/apdcephfs_cq8/share_2992679/lora_sdxl" \
  --resolution=1024  \
  --train_batch_size=2 \
  --max_train_steps=20000 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --validation_prompt="a human face" \
  --checkpointing_steps=2500 \
  --image_folder="/apdcephfs_cq8/share_2992679/Datasets/all_frames/" \
  --validation_epochs=100000 --rank=16
