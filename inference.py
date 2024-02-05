import os, cv2
import torch
import argparse
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder

from svd_poseguider.models.controlnet_svd import ControlNetSVDModel
from svd_poseguider.models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from svd_poseguider.pipelines.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
from svd_poseguider.utils import save_gifs_side_by_side, load_images_from_folder_to_pil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to inference SVD PoseGuider."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        required=True,
        help="",
    )
    parser.add_argument(
        "--validation_folder",
        type=str,
        default=None,
        required=True,
        help="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )

    args = parser.parse_args()
    return args

# Main script
if __name__ == "__main__":
    args = parse_args()
    # Load validation images and control images
    args.validation_image_folder = os.path.join(args.validation_folder, "image")
    args.validation_control_folder = os.path.join(args.validation_folder, "dwpose")
    args.validation_image = os.path.join(args.validation_folder, "image/000000.png")
    validation_images = load_images_from_folder_to_pil(args.validation_image_folder, target_size=[args.height, args.width])
    validation_control_images = load_images_from_folder_to_pil(args.validation_control_folder, target_size=[args.height, args.width])
    validation_image = Image.open(args.validation_image).convert('RGB')

    # Load and set up the pipelines
    controlnet = ControlNetSVDModel.from_pretrained(args.controlnet_path, subfolder="controlnet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path,
                                                                  subfolder="image_encoder", 
                                                                  variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path,
                                                       subfolder="vae", 
                                                       variant="fp16")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args.pretrained_model_name_or_path,
                                                            subfolder="unet",
                                                            variant="fp16")
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(args.pretrained_model_name_or_path,
                                                                      controlnet=controlnet,
                                                                      unet=unet,
                                                                      image_encoder=image_encoder,
                                                                      vae=vae,
                                                                      torch_dtype=torch.float16)
    
    pipeline.enable_model_cpu_offload()
    # pipeline.enable_xformers_memory_efficient_attention()

    # Create output directory if it doesn't exist
    val_save_dir = os.path.join(args.output_dir, "validation_images")
    os.makedirs(val_save_dir, exist_ok=True)

    # Inference and saving loop
    video_frames = pipeline(
        validation_image, 
        validation_control_images[:14],
        height=args.height,
        width=args.width,
        decode_chunk_size=8,
        num_frames=14,
        motion_bucket_id=127,
        controlnet_cond_scale=0.9,
        noise_aug_strength=0).frames

    save_gifs_side_by_side(video_frames, 
                           validation_images, 
                           validation_control_images,
                           val_save_dir)
