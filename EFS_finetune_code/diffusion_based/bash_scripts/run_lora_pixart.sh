export MODEL_NAME="/apdcephfs/tmp/PixArt-XL-2-1024-MS"

accelerate launch --mixed_precision="no" train_scripts/train_pixart_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="/apdcephfs_cq10/pixart_CelebDF" \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=20000 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompt="a human face" \
  --checkpointing_steps=2500 \
  --image_folder="/apdcephfs_cq10/Datasets/Celeb-DF-v2/all_frames/" \
  --validation_epochs=1000 --rank=16

accelerate launch --mixed_precision="no" train_scripts/train_pixart_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir="/apdcephfs_cq10/pixart_FF" \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=20000 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompt="a human face" \
  --checkpointing_steps=2500 \
  --image_folder="/apdcephfs_cq10/Datasets/FaceForensics++/all_frames/" \
  --validation_epochs=1000 --rank=16

