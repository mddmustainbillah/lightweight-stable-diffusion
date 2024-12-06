import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from utils import prepare_image

def test(input_image_path, prompt, checkpoint_path):
    # Load fine-tuned model
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32
    )
    
    # Load and prepare input image
    init_image = Image.open(input_image_path).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Generate image
    output = pipeline(
        prompt=prompt,
        image=init_image,
        strength=0.75,
        guidance_scale=7.5
    ).images[0]
    
    # Save output
    output.save("output_image.png")

if __name__ == "__main__":
    test(
        input_image_path="images/mustain_photo1.JPEG",
        prompt="A professional portrait photo, highly detailed face, studio lighting, 8k uhd, sharp focus",
        checkpoint_path="checkpoints/epoch_5"
    ) 