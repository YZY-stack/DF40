from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch
import numpy as np
from torchvision.utils import make_grid
import os
from PIL import Image
from peft import LoraConfig

prior_pipeline = DiffusionPipeline.from_pretrained("/apdcephfs_cq8/share_2992679/private/chengmingxu/model_weights/kandinsky-2-2-prior", torch_dtype=torch.float16)
# prior_pipeline.load_attn_procs('/apdcephfs_cq8/share_2992679/private/chengmingxu/wxpay/kandinsky_2.2_prior/')
prior_components = {"prior_" + k: v for k,v in prior_pipeline.components.items()}
pipe = AutoPipelineForText2Image.from_pretrained("/apdcephfs_cq8/share_2992679/private/chengmingxu/model_weights/kandinsky-2-2-decoder/", 
                                                 **prior_components,
                                                 torch_dtype=torch.float16)
# pipe.unet.load_attn_procs('/apdcephfs_cq8/share_2992679/private/chengmingxu/wxpay/kandinsky_2.2_decoder/')
# pipe.prior_prior.load_attn_procs('/apdcephfs_cq8/share_2992679/private/chengmingxu/wxpay/kandinsky_2.2_prior/')
# pipe.enable_model_cpu_offload()

# pipe.load_lora_weights('/apdcephfs_cq8/share_2992679/private/chengmingxu/wxpay/kandinsky_2.2_decoder/')

prior_lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

unet_lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

pipe.prior_prior.requires_grad_(False)
pipe.unet.requires_grad_(False)
pipe.prior_prior.add_adapter(prior_lora_config)
pipe.unet.add_adapter(unet_lora_config)

from safetensors.torch import load_file
def load_lora_weights(model, file_path):
    prior_lora_weights=load_file(file_path)
    prior_lora_weights_new = {}
    for k, v in prior_lora_weights.items():
        prior_lora_weights_new[k.replace('unet.', '')[:-7]+'.default.weight'] = v
    model.load_state_dict(prior_lora_weights_new, strict=False)
    lora_layers = filter(lambda p: p.requires_grad, model.parameters()) 
    for param in lora_layers:
        param.data = param.to(torch.float32)

prior_file_path = '/apdcephfs_cq8/share_2992679/private/chengmingxu/wxpay/kandinsky_2.2_prior/checkpoint-5000/pytorch_lora_weights.safetensors'
load_lora_weights(pipe.prior_prior, prior_file_path)

unet_file_path = '/apdcephfs_cq8/share_2992679/private/chengmingxu/wxpay/kandinsky_2.2_decoder/pytorch_lora_weights.safetensors'
load_lora_weights(pipe.unet, unet_file_path)

# prior_lora_layers = filter(lambda p: p.requires_grad, pipe.prior_prior.parameters()) 
# unet_lora_layers = filter(lambda p: p.requires_grad, pipe.unet.parameters())
# for param in prior_lora_layers:
#     param.data = param.to(torch.float32)
# for param in unet_lora_layers:
#     param.data = param.to(torch.float32)

# import pdb;pdb.set_trace()
    
pipe = pipe.to(device='cuda')

prompt='a wxpay human face'
negative_prompt = 'ugly'
# image = pipe(prompt=prompt).images[0]
# image.save("robot_pokemon.png")

prompt_images = []
for _ in range(10):
    prompt_images.append(
        torch.tensor(np.asarray(pipe(prompt=prompt, negative_prompt=negative_prompt, prior_guidance_scale =1.0, height=512, width=512).images[0])).permute(2, 0, 1)
    )

out_path = './outputs/kandinsky_decoder'
os.makedirs(out_path, exist_ok=True)
image_grid = Image.fromarray(make_grid(prompt_images, nrow=5).permute(1, 2, 0).numpy())
image_grid.save(os.path.join(out_path, 'test_grid.png'))