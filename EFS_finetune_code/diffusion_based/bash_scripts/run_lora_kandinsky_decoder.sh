export PRIOR_MODEL_NAME="/apdcephfs_cq8/share_2992679/model_weights/kandinsky-2-2-prior/"
export DECODER_MODEL_NAME="/apdcephfs_cq8/share_2992679/model_weights/kandinsky-2-2-decoder/"

accelerate launch --mixed_precision="fp16" train_scripts/train_kandinsky_decoder_lora.py \
  --pretrained_prior_model_name_or_path=$PRIOR_MODEL_NAME \
  --pretrained_decoder_model_name_or_path=$DECODER_MODEL_NAME \
  --output_dir="/apdcephfs_cq8/share_2992679/chengmingxu/kandinsky_2.2_decoder" \
  --resolution=768 \
  --train_batch_size=8 \
  --max_train_steps=10000 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompt="a human face" \
  --checkpointing_steps=10000 \
  --image_folder="/apdcephfs_cq8/share_2992679/Datasets/all_frames/" \
  --validation_epochs=1000
