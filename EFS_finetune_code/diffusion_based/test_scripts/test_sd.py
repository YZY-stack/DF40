# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile

import safetensors
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, AutoencoderKL  # noqa: E402

import argparse
import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

args = parse_args()

if args.vae_path is not None:
    vae = AutoencoderKL.from_pretrained(
        args.vae_path,
        subfolder="vae",
    )
    vae = vae.to(torch.float16)


    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
else:
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)   
pipeline = pipeline.to('cuda')

# load attention processors
# pipeline.set_progress_bar_config(disable=True)

prompt = 'A serious capybara at work, wearing a suit'
negative_prompt = ''

prompt_images = []
for _ in range(4):
    prompt_images.append(
        torch.tensor(np.asarray(pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=20,
                    num_images_per_prompt=1, guidance_scale=7.5, height=args.resolution, width=args.resolution).images[0])).permute(2, 0, 1)
    )
os.makedirs(args.out_path, exist_ok=True)
image_grid = Image.fromarray(make_grid(prompt_images, nrow=2).permute(1, 2, 0).numpy())
image_grid.save(os.path.join(args.out_path, 'test_grid.png'))