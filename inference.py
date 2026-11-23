from pipelines.pipeline import IMAGGarment
import os
import torch

from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from adapter.attention_processor import LoraLogoCacheSAttnProcessor2_0,LogoCacheSAttnProcessor2_0, LogoRefSAttnProcessor2_0, LogoCacheCAttnProcessor2_0 , SKIPAttnProcessor2_0
import argparse
from adapter.resampler import Resampler,ImageProjModel
from train import get_masked_image

def resize_img(input_image, max_side=640, min_side=512, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    return input_image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    max_w,max_h=0,0
    for img in imgs :
        max_w = max(max_w,img.size[0])
        max_h = max(max_h,img.size[1])
            
    w, h = max_w,max_h
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def prepare(args):
    generator = torch.Generator(device=args.device).manual_seed(42)
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path).to(dtype=torch.float16, device=args.device)
    
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(
        dtype=torch.float16,device=args.device)
    
    # set attention processor
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
        if cross_attention_dim is None:
            if 'up_blocks.0.attentions.0' in name:
                attn_procs[name] = LogoRefSAttnProcessor2_0(name, hidden_size,skip_sketch=False,skip_color=True,skip_logo=False)
            elif 'up_blocks.0.attentions.1' in name:
                attn_procs[name] = LogoRefSAttnProcessor2_0(name, hidden_size,skip_sketch=True,skip_color=False,skip_logo=False)
            else:
                attn_procs[name] = LogoRefSAttnProcessor2_0(name, hidden_size,skip_sketch=True,skip_color=True,skip_logo=False)
        else:
            attn_procs[name] = SKIPAttnProcessor2_0(name, hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules = adapter_modules.to(dtype=torch.float16, device=args.device)
    del st

    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(
        dtype=torch.float16,
        device=args.device)
    attn_procs2 = {}
    st = ref_unet.state_dict()
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
            attn_procs2[name] = LoraLogoCacheSAttnProcessor2_0(name, hidden_size,skip_sketch=skip_sketch,skip_color=skip_color)
        else:
            attn_procs2[name] = SKIPAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)  # .to(accelerator.device)]
    ref_unet.set_attn_processor(attn_procs2)
    # ref_unet.set_attn_processor(
    #     {name: LogoCacheSAttnProcessor2_0() for name in ref_unet.attn_processors.keys()})  # set cache
    del st
    ref_unet.to(dtype=torch.float16,device=args.device)
    # weights load
    model_sd = torch.load(args.model_ckpt, map_location="cpu")["module"]

    ref_unet_dict = {}
    unet_dict = {}
    adapter_modules_dict = {}
    for k in model_sd.keys():
        if k.startswith("ref_unet"):
            ref_unet_dict[k.replace("ref_unet.", "")] = model_sd[k]
        elif k.startswith("unet"):
            unet_dict[k.replace("unet.", "")] = model_sd[k]
        elif k.startswith("adapter_modules"):
            adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
        else:
            # print(k)
            continue

    ref_unet.load_state_dict(ref_unet_dict,strict = False)
    adapter_modules.load_state_dict(adapter_modules_dict)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = IMAGGarment(unet=unet, reference_unet=ref_unet, vae=vae, 
                        # tokenizer=tokenizer,
                        # tokenizer_2 = tokenizer_2,
                        #   text_encoder=text_encoder, 
                        # text_encoder_2 = text_encoder_2,
                         scheduler=noise_scheduler,
                         safety_checker=StableDiffusionSafetyChecker,
                         feature_extractor=CLIPImageProcessor)
    return pipe, generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a garment from visual conditions")
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="SDXL model path or Hugging Face model ID")
    parser.add_argument("--vae_model_name_or_path", default="madebyollin/sdxl-vae-fp16-fix",
                        help="SDXL VAE path or Hugging Face model ID")
    parser.add_argument("--model_ckpt", required=True, help="Trained IMAGGarment+ checkpoint")
    parser.add_argument("--sketch_path", required=True)
    parser.add_argument("--logo_path", required=True)
    parser.add_argument("--color_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--cloth_path", default=None, help="Optional ground truth image for the comparison grid")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    pipe, generator = prepare(args)
    img_transform = transforms.Compose([
        transforms.Resize([args.height, args.width], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    sketch_img = get_masked_image(args.mask_path, args.sketch_path).resize((args.width, args.height))
    logo_img = Image.open(args.logo_path).convert("RGB").resize((args.width, args.height))
    color_img = Image.open(args.color_path).convert("RGB").resize((args.width, args.height))
    mask_img = Image.open(args.mask_path).convert("RGB").resize((args.width, args.height))

    result = pipe(
        ref_image=img_transform(sketch_img).unsqueeze(0),
        ref_logo=img_transform(logo_img).unsqueeze(0),
        ref_color=img_transform(color_img).unsqueeze(0),
        width=args.width,
        height=args.height,
        num_images_per_prompt=1,
        guidance_scale=2.5,
        image_scale=1.0,
        logo_scale=1.0,
        color_scale=1.0,
        lora_scale=1.0,
        generator=generator,
        num_inference_steps=50,
    ).images[0]

    name = os.path.splitext(os.path.basename(args.sketch_path))[0] + ".png"
    single_dir = os.path.join(args.output_path, "single")
    grid_dir = os.path.join(args.output_path, "all")
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)
    result.save(os.path.join(single_dir, name))

    comparison = [color_img, mask_img]
    if args.cloth_path:
        comparison.append(Image.open(args.cloth_path).convert("RGB").resize((args.width, args.height)))
    comparison.extend([logo_img, sketch_img, result])
    image_grid(comparison, 1, len(comparison)).save(os.path.join(grid_dir, name))
