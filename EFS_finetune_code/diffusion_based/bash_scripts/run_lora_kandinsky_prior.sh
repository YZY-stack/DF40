export MODEL_NAME="/apdcephfs_cq8/share_2992679/model_weights/kandinsky-2-2-prior/"

accelerate launch --mixed_precision="fp16" train_scripts/train_kandinsky_prior_lora.py \
  --pretrained_prior_model_name_or_path=$MODEL_NAME \
  --output_dir="/apdcephfs_cq8/share_2992679/kandinsky_2.2_prior" \
  --resolution=512 \
  --train_batch_size=16\
  --max_train_steps=20000 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompt="a human face" \
  --checkpointing_steps=2500 \
  --image_folder="/apdcephfs_cq8/share_2992679/Datasets/all_frames/" \
  --validation_epochs=1000

