import torch
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL, Transformer2DModel
from peft import PeftModel
import numpy as np
from PIL import Image
import os
from torchvision.utils import make_grid

# You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-512x512" too.
# pipe = PixArtAlphaPipeline.from_pretrained("/apdcephfs/private_chengmingxu/tmp/PixArt-XL-2-1024-MS/", torch_dtype=torch.float16, use_safetensors=True)

# If use DALL-E 3 Consistency Decoder
# pipe.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)

# If use SA-Solver sampler
# from diffusion.sa_solver_diffusers import SASolverScheduler
# pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')

# If loading a LoRA model
transformer = Transformer2DModel.from_pretrained("/apdcephfs/private_chengmingxu/tmp/PixArt-XL-2-1024-MS/", subfolder="transformer", torch_dtype=torch.float16)
# transformer = PeftModel.from_pretrained(transformer, "/apdcephfs_cq10/share_1275017/chengmingxu/wxpay/pixart_FF/checkpoint-20000")
transformer = PeftModel.from_pretrained(transformer, "/apdcephfs_cq10/share_1275017/chengmingxu/wxpay/pixart_CelebDF/checkpoint-20000")
pipe = PixArtAlphaPipeline.from_pretrained("/apdcephfs/private_chengmingxu/tmp/PixArt-XL-2-1024-MS/", transformer=transformer, torch_dtype=torch.float16, use_safetensors=True)
del transformer

# Enable memory optimizations.
pipe.enable_model_cpu_offload()

prompt = "a wxpay human face"
negative_prmopt = ''

out_path = "/apdcephfs_cq10/share_1275017/chengmingxu/wxpay_images/pixart_CelebDF"
frame_path = "/apdcephfs_cq10/share_1275017/chengmingxu/Datasets/wxpay/Celeb-DF-v2/all_frames"
# out_path = "/apdcephfs_cq10/share_1275017/chengmingxu/wxpay_images/pixart_FF"
# frame_path = "/apdcephfs_cq10/share_1275017/chengmingxu/Datasets/wxpay/FaceForensics++ 2/all_frames"

generator = torch.Generator(device='cuda')
os.makedirs(out_path, exist_ok=True)
for seed, frame_name in enumerate(os.listdir(frame_path)):
# for seed in range(25):
    print(frame_name)
    if os.path.isfile(os.path.join(out_path, frame_name)): continue
    generator = generator.manual_seed(seed)

    image = pipe(prompt, negative_prmopt=negative_prmopt).images[0]
    image.save(os.path.join(out_path, frame_name))