export MODEL_NAME="/apdcephfs/tmp/ddpm-celebahq-256"

accelerate launch --mixed_precision="fp16" train_scripts/train_ddpm.py  \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="/apdcephfs_cq10/ddpm_CelebDF" \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=8 \
  --max_train_steps=20000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --validation_prompt="a human face" \
  --checkpointing_steps=10000 \
  --image_folder="/apdcephfs_cq10/share_1275017/Datasets/Celeb-DF-v2/all_frames/" \
  --validation_epochs=10000

accelerate launch --mixed_precision="fp16" train_scripts/train_ddpm.py  \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="/apdcephfs_cq10/ddpm_FF" \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=8 \
  --max_train_steps=20000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --enable_xformers_memory_efficient_attention \
  --validation_prompt="a human face" \
  --checkpointing_steps=10000 \
  --image_folder="/apdcephfs_cq10/Datasets/FaceForensics++/all_frames/" \
  --validation_epochs=10000

