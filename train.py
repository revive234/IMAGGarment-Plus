

import argparse
import logging
import time
import itertools
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from adapter.resampler import Resampler,ImageProjModel
# from Logo_dataset import FGGDataset, collate_fn
from adapter.attention_processor import RefSAttnProcessor2_0,CacheAttnProcessor2_0,SKIPAttnProcessor2_0,CAttnProcessor2_0,LogoCacheSAttnProcessor2_0,  SKIPAttnProcessor2_0, LogoRefSAttnProcessor2_0,LogoCacheCAttnProcessor2_0,LoraLogoCacheSAttnProcessor2_0

logger = get_logger(__name__)


import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
import os
from random import choice
from PIL import Image, ImageFilter
import numpy as np
def get_masked_image(mask_path,sketch_path):
    image = Image.open(sketch_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size)   
    image_np = np.array(image)      # [H, W, 3]
    mask_np = np.array(mask)        # [H, W], 值范围 [0,255]

    # 步骤 1：反转 mask（黑白对调）
    inverted_mask = 255 - mask_np  # 白变黑，黑变白

    # 步骤 2：归一化 mask 为 0~1，再扩展为 RGB 通道
    mask_norm = (inverted_mask / 255.0).astype(np.float32)  # [H, W]
    mask_rgb = np.stack([mask_norm]*3, axis=-1)             # [H, W, 3]

    # 步骤 3：乘 image，屏蔽掉原来的白色区域
    masked_image = (image_np * mask_rgb).astype(np.uint8)

    # 步骤 4：把原来白色区域（现在是黑色）填为灰色
    gray_color = np.array([128, 128, 128], dtype=np.uint8)
    white_area = mask_np == 255
    masked_image[white_area] = gray_color

    # 保存结果
    result = Image.fromarray(masked_image)
    return result
class FGGDataset(Dataset):
    def __init__(
            self,
            json_file,
            # tokenizer,
            # tokenizer_2,
            center_crop = True,
            size=[640,512],
            image_root_path="",
    ):

        with open(json_file, 'r') as file:
            self.data = json.load(file)

        print('=========', len(self.data),size)

        # self.tokenizer = tokenizer
        self.size = size
        self.image_root_path = image_root_path
        self.center_crop = center_crop
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # self.tokenizer = tokenizer
        # self.tokenizer_2 = tokenizer_2
        # self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # text = item["caption"]
        cloth_path = os.path.join(self.image_root_path,item["cloth"])
        cloth_img = Image.open(cloth_path).convert("RGB")
        r = random.random()
        if r < 1/2:
            sketch_path = os.path.join(self.image_root_path,'sketch3',item["sketch"].split('/')[-1])
        elif r < 4/5:
            sketch_path = os.path.join(self.image_root_path,'canny_edge3',item["sketch"].split('/')[-1])
        else:
            sketch_path = os.path.join(self.image_root_path,'photosketch3',item["sketch"].split('/')[-1])
        # sketch_path = os.path.join(self.image_root_path,'pidinet',item["sketch"].split('/')[-1])

        mask_path=os.path.join(self.image_root_path,item['mask'])
        mask_img = Image.open(mask_path)

        # sketch_img = Image.open(sketch_path).convert("RGB")
        sketch_img = get_masked_image(mask_path,sketch_path)

        logo_path=os.path.join(self.image_root_path,item['logo'])
        logo_img = Image.open(logo_path).convert("RGB")
        
        
        color_path = os.path.join(self.image_root_path,item['color'])
        color_img = Image.open(color_path).convert("RGB")

        # original size
        original_width, original_height = cloth_img.size
        original_size = torch.tensor([original_height, original_width]) 



        drop_sketch_embed = 0
        drop_logo_embed = 0
        drop_mask_embed = 0
        drop_color_embed = 0

        rand_num = random.random()
        if rand_num < 0.05:
            drop_sketch_embed = 1
        elif rand_num < 0.1:  # 0.55: #0.1:
            drop_logo_embed = 1
        elif rand_num < 0.15:  # 0.6: #0.15:
            drop_mask_embed = 1
        elif rand_num < 0.20:
            drop_color_embed = 1
        elif rand_num < 0.25:
            drop_sketch_embed = 1
            drop_logo_embed = 1
            drop_mask_embed = 1
            drop_color_embed = 1

        vae_cloth = self.transform(cloth_img)
        vae_sketch = self.transform(sketch_img)
        vae_logo = self.transform(logo_img)
        vae_color = self.transform(color_img)
        # clip_image = self.clip_image_processor(images=mask_img, return_tensors="pt").pixel_values

        # random crop
        delta_h = vae_cloth.shape[1] - self.size[0]
        delta_w = vae_cloth.shape[2] - self.size[1]

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            vae_cloth, top=top, left=left, height=self.size[0], width=self.size[1]
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        # get text and tokenize
        # text_input_ids = self.tokenizer(
        #     text,
        #     max_length=self.tokenizer.model_max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # ).input_ids
        
        # text_input_ids_2 = self.tokenizer_2(
        #     text,
        #     max_length=self.tokenizer_2.model_max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # ).input_ids


        return {
            "vae_cloth": image,
            "vae_sketch": vae_sketch,
            "vae_logo":vae_logo,
            "vae_color":vae_color,
            # "clip_image": clip_image,
            "drop_sketch_embed":drop_sketch_embed,
            "drop_logo_embed": drop_logo_embed,
            "drop_mask_embed": drop_mask_embed,
            "drop_color_embed": drop_color_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size[0], self.size[1]]),
            # "text_input_ids": text_input_ids,
            # "text_input_ids_2": text_input_ids_2,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    vae_cloth = torch.stack([example["vae_cloth"] for example in data]).to(
        memory_format=torch.contiguous_format).float()
    vae_sketch = torch.stack([example["vae_sketch"] for example in data]).to(
        memory_format=torch.contiguous_format).float()

    vae_logo = torch.stack([example["vae_logo"] for example in data]).to(
        memory_format=torch.contiguous_format).float()
    
    vae_color = torch.stack([example["vae_color"] for example in data]).to(
        memory_format=torch.contiguous_format).float()
    
    # clip_image = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_sketch_embed = [example["drop_sketch_embed"] for example in data]
    drop_logo_embed = [example["drop_logo_embed"] for example in data]
    drop_color_embed = [example["drop_color_embed"] for example in data]
    drop_mask_embed = [example["drop_mask_embed"] for example in data]

    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    # text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    # text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)

    # text = [example["text"] for example in data]
    # input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    # null_input_ids = torch.cat([example["null_text_input_ids"] for example in data], dim=0)

    return {
        "vae_cloth": vae_cloth,
        "vae_sketch": vae_sketch,
        "vae_logo":vae_logo,
        "vae_color":vae_color,
        # "clip_image": clip_image,
        "drop_sketch_embed": drop_sketch_embed,
        "drop_logo_embed":drop_logo_embed,
        "drop_color_embed":drop_color_embed,
        "drop_mask_embed":drop_mask_embed,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        # "text_input_ids": text_input_ids,
        # "text_input_ids_2": text_input_ids_2,

    }




def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Path or Hugging Face model ID for the SDXL VAE.",
    )
    parser.add_argument(
        "--pretrained_text_model_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    parser.add_argument(
        "--dataset_json_path",
        type=str,
        required=True,
        help="Path to dataset json file.",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        required=True,
        help="Path of dataset.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        '--clip_penultimate',
        type=bool,
        default=False,
        help='Use penultimate CLIP layer for text embedding'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0.05, help="noise_offset."
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=0, help="noise_offset."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Desired width of generated image",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Desired height of generated image",
    )
    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(model,load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    # _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
    # epoch = checkpoint_state_dict["epoch"]
    # last_global_step = checkpoint_state_dict["last_global_step"]
    # del checkpoint_state_dict

    state_dict = torch.load(load_dir, map_location='cpu')
    model.load_state_dict(state_dict['module'], strict=False)
    epoch = state_dict['epoch']
    last_global_step  = state_dict['global_steps']
    return (epoch, last_global_step)


def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


class SDModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, unet, ref_unet, adapter_modules) -> None:
        super().__init__()

        self.unet = unet
        self.ref_unet = ref_unet
        self.adapter_modules = adapter_modules

    def forward(self, encoder_hidden_states, latents, ref_latents, timesteps,unet_added_cond_kwargs):
        ref_timesteps = torch.zeros_like(timesteps)
        ref_timesteps = torch.cat([ref_timesteps,ref_timesteps,ref_timesteps],dim=0)
        ref_unet_added_cond_kwargs = {
             "text_embeds": torch.cat([unet_added_cond_kwargs['text_embeds'],unet_added_cond_kwargs['text_embeds'],unet_added_cond_kwargs['text_embeds']],dim=0),
             "time_ids": torch.cat([unet_added_cond_kwargs['time_ids'],unet_added_cond_kwargs['time_ids'],unet_added_cond_kwargs['time_ids']],dim=0),
        }
        _ = self.ref_unet(
            ref_latents,
            ref_timesteps,
            None,
            added_cond_kwargs=ref_unet_added_cond_kwargs,
            return_dict=False,
        )
        
        # get cache tensors
        sa_hidden_states = {}
        for name in self.ref_unet.attn_processors.keys():
            if "attn1" in  name:
                sa_hidden_states[name] = self.ref_unet.attn_processors[name].cache["hidden_states"]

        # get noise predictions
        # Predict the noise residual and compute loss
        noise_pred = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={
                "sa_hidden_states": sa_hidden_states,
            },
            added_cond_kwargs=unet_added_cond_kwargs
        ).sample

        return noise_pred


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    # tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    # tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    # text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path)
    

    # set attention processor
    # denosing unet's attention processor
    attn_procs = {}
    st = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        # lora_rank = hidden_size // 2 # args.lora_rank
        if cross_attention_dim is None:
            layer_name = name.split('.processor')[0]  
            if 'up_blocks.0.attentions.0' in name:
                attn_procs[name] = LogoRefSAttnProcessor2_0(name, hidden_size,skip_sketch=False,skip_color=True,skip_logo=False)
                weights = {
                "to_k_ref.weight": st[layer_name + ".to_k.weight"],
                "to_v_ref.weight": st[layer_name + ".to_v.weight"],
                "to_k_ref_logo.weight": st[layer_name + ".to_k.weight"],
                "to_v_ref_logo.weight": st[layer_name + ".to_v.weight"],
            }
            elif 'up_blocks.0.attentions.1' in name:
                attn_procs[name] = LogoRefSAttnProcessor2_0(name, hidden_size,skip_sketch=True,skip_color=False,skip_logo=False)
                weights = {
                "to_k_ref_logo.weight": st[layer_name + ".to_k.weight"],
                "to_v_ref_logo.weight": st[layer_name + ".to_v.weight"],
                "to_k_ref_color.weight": st[layer_name + ".to_k.weight"],
                "to_v_ref_color.weight": st[layer_name + ".to_v.weight"],
            }
            else:
                attn_procs[name] = LogoRefSAttnProcessor2_0(name, hidden_size,skip_sketch=True,skip_color=True,skip_logo=False)
                weights = {
                    "to_k_ref_logo.weight": st[layer_name + ".to_k.weight"],
                    "to_v_ref_logo.weight": st[layer_name + ".to_v.weight"],
                }
            attn_procs[name].load_state_dict(weights)
        else:
            attn_procs[name] = SKIPAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)  # .to(accelerator.device)]
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    del st

    # reference unet's attention processor
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    attn_procs2 = {}
    st = ref_unet.state_dict()
    # ref_unet.set_attn_processor(
    #     {name: CacheAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache

    non_sketch_layer = ['up_blocks.0.attentions.1','up_blocks.0.attentions.2','up_blocks.1.attentions.0','up_blocks.1.attentions.1','up_blocks.1.attentions.2']
    non_color_layer = ['up_blocks.0.attentions.2','up_blocks.1.attentions.0','up_blocks.1.attentions.1','up_blocks.1.attentions.2']
    for name in ref_unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else ref_unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = ref_unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(ref_unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = ref_unet.config.block_out_channels[block_id]
        # lora_rank = hidden_size // 2 # args.lora_rank
        if cross_attention_dim is None:
            current_layer = ".".join(name.split('.')[:-4])
            skip_sketch, skip_color = False, False
            if current_layer in non_sketch_layer:
                skip_sketch = True
                print(current_layer,'skip_sketch:',skip_sketch)
            if current_layer in non_color_layer:
                skip_color = True
                print(current_layer,'skip_color:',skip_color)
            attn_procs2[name] = LoraLogoCacheSAttnProcessor2_0(name, hidden_size,rank=128,skip_sketch=skip_sketch,skip_color=skip_color)
        else:
            # attn_procs2[name] = SKIPAttnProcessor2_0(name, hidden_size=hidden_size,
            #                                      cross_attention_dim=cross_attention_dim) 
            attn_procs2[name] = SKIPAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)

            
    ref_unet.set_attn_processor(attn_procs2)

    del st
    
    
    # sef freeze and trainable
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    # text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    # image_encoder.requires_grad_(False)
    # image_proj.requires_grad_(False)
    ref_unet.requires_grad_(False)
    adapter_modules.requires_grad_(False)
    ref_unet_dict={}
    for name, param in ref_unet.named_parameters():
        if  'lora' in name :
            last_name = '.'.join(name.split('.')[-3:-1])
            # print(name,last_name)
            if last_name not in ref_unet_dict:
                ref_unet_dict[last_name] = 0
            ref_unet_dict[last_name] = ref_unet_dict[last_name] + 1
            param.requires_grad = True
    for key,value in ref_unet_dict.items():
        print(f'{key}:{value}')

    adapter_modules_dict = {}
    for name, param in adapter_modules.named_parameters():
        if  'to_k_ref_color' in name or 'to_v_ref_color' in name or 'to_k_ref_logo' in name or 'to_v_ref_logo' in name or 'to_k_ref' in name or 'to_v_ref' in name:
            last_name = name.split('.')[-2]
            # print(name,last_name)
            if last_name not in adapter_modules_dict:
                adapter_modules_dict[last_name] = 0
            adapter_modules_dict[last_name] = adapter_modules_dict[last_name] + 1
            param.requires_grad = True

    for key,value in adapter_modules_dict.items():
        print(f'{key}:{value}')
    sd_model = SDModel(unet, ref_unet, adapter_modules)
    


    
    params_to_opt = itertools.chain( sd_model.adapter_modules.parameters(),sd_model.ref_unet.parameters()
                                )


    accelerator.print("Trainable parameters:  ref_unet:{:.2f}M, adapter_modules:{:.2f}M, unet:{:.2f}M".format(
        count_model_params(sd_model.ref_unet),
        count_model_params(sd_model.adapter_modules),
        count_model_params(sd_model.unet)))
    # accelerator.print("Trainable parameters: {:.2f}M".format(len(params_to_opt)))
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    if (
            accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        # use deepspeed config
        optimizer = DummyOptim(
            params_to_opt,
            lr=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"],
            weight_decay=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["weight_decay"]
        )

    # TODO (patil-suraj): load scheduler using args
    noise_scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing", prediction_type="epsilon",
    )

    dataset = FGGDataset(
        
        args.dataset_json_path,
        # tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        image_root_path= args.image_root_path,
        size = [args.height,args.width]
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=0,
    )

    if accelerator.state.deepspeed_plugin is not None:
        # here we use agrs.gradient_accumulation_steps
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        # use deepspeed scheduler
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_num_steps"]
        )

    if (
            accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto"
    ):
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    sd_model, optimizer, lr_scheduler = accelerator.prepare(sd_model, optimizer, lr_scheduler)
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin is None:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        if accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]:
            weight_dtype = torch.float16
        elif accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]:
            weight_dtype = torch.bfloat16
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    # text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)
    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
   
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        last_epoch, last_global_step = load_training_checkpoint(
            sd_model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        starting_epoch = last_epoch
        global_steps = last_global_step



    for epoch in range(starting_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        step = 0
        begin = time.perf_counter()
        for batch in train_dataloader:
            load_data_time = time.perf_counter() - begin
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(
                    batch["vae_cloth"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                ref_latents = vae.encode(
                    batch["vae_sketch"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                ref_latents = ref_latents * vae.config.scaling_factor

                logo_latents=vae.encode(
                    batch["vae_logo"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                logo_latents = logo_latents * vae.config.scaling_factor
                
                color_latents=vae.encode(
                    batch["vae_color"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                color_latents = color_latents * vae.config.scaling_factor
  
                
                # image_embeds = image_encoder(batch['clip_image'].to(accelerator.device, dtype=weight_dtype),
                #                              output_hidden_states=True).image_embeds
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if args.noise_offset > 0:
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                )
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # image_embeds_ = []
            # for image_embed, drop_image_embed in zip(image_embeds, batch["drop_mask_embed"]):
            #         if drop_image_embed == 1:
            #             image_embeds_.append(torch.zeros_like(image_embed))
            #         else:
            #             image_embeds_.append(image_embed)
            # image_embeds = torch.stack(image_embeds_)

            ref_latents_ = []
            for ref_latent, drop_sketch_embed in zip(ref_latents, batch['drop_sketch_embed']):
                if drop_sketch_embed == 1:
                    ref_latents_.append(torch.zeros_like(ref_latent))
                else:
                    ref_latents_.append(ref_latent)
            ref_latents = torch.stack(ref_latents_)

            ref_logo_latents_ = []
            for ref_latent, drop_logo_embed in zip(logo_latents, batch['drop_logo_embed']):
                if drop_logo_embed == 1:
                    ref_logo_latents_.append(torch.zeros_like(ref_latent))
                else:
                    ref_logo_latents_.append(ref_latent)
            logo_latents = torch.stack(ref_logo_latents_)


            ref_color_latents_ = []
            for ref_latent, drop_color_embed in zip(color_latents, batch['drop_color_embed']):
                if drop_color_embed == 1:
                    ref_color_latents_.append(torch.zeros_like(ref_latent))
                else:
                    ref_color_latents_.append(ref_latent)
            color_latents = torch.stack(ref_color_latents_)

            # 获取文本嵌入（SDXL使用两个文本编码器）
            # with torch.no_grad():
            #     encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
            #     text_embeds = encoder_output.hidden_states[-2]
            #     encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
            #     pooled_text_embeds = encoder_output_2[0]
            #     text_embeds_2 = encoder_output_2.hidden_states[-2]
            #     text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)  # 连接两个文本编码器的输出


            # with torch.no_grad():
            #     encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    latents, noise, timesteps
                )
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.prediction_type}"
                )
            
            ref_latents=torch.cat([ref_latents,logo_latents,color_latents],dim=0)

            # 添加SDXL所需的额外条件（图像尺寸和裁剪信息）
            add_time_ids = [
                batch["original_size"].to(accelerator.device),
                batch["crop_coords_top_left"].to(accelerator.device),
                batch["target_size"].to(accelerator.device),
            ]
            add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
            unet_added_cond_kwargs = { "text_embeds": torch.zeros(add_time_ids.shape[0], 1280,device = accelerator.device),"time_ids": add_time_ids}
            model_pred = sd_model(None, noisy_latents, ref_latents, timesteps,unet_added_cond_kwargs)

            if args.snr_gamma == 0:

                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                )
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="none"
                )
                loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                )
                loss = loss.mean()
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item()

            # Backpropagate
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # do nothing
                lr_scheduler.step()  # only for not deepspeed lr_scheduler
                optimizer.zero_grad()  # do nothing

                if accelerator.sync_gradients:
                    accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_steps)
                    train_loss = 0.0

            if accelerator.is_main_process:
                logging.info(
                    "Epoch {}, step {},  step_loss: {}, lr: {}, time: {}, data_time: {}".format(
                        epoch, global_steps, loss.detach().item(), lr_scheduler.get_lr()[0],
                        time.perf_counter() - begin, load_data_time)
                )
            global_steps += 1
            step += 1

            # checkpoint
            if isinstance(checkpointing_steps, int):
                if global_steps % checkpointing_steps == 0:
                    checkpoint_model(args.output_dir, global_steps, sd_model, epoch, global_steps)

            # stop training
            if global_steps >= args.max_train_steps:
                break
            begin = time.perf_counter()

    accelerator.wait_for_everyone()
    # Save last model
    checkpoint_model(args.output_dir, global_steps, sd_model, epoch, global_steps)

    accelerator.end_training()


if __name__ == "__main__":
    main()
