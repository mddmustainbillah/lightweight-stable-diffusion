import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from utils import prepare_image
from huggingface_hub import HfFolder, hf_hub_download, snapshot_download
from PIL import Image
import json
from pathlib import Path
from dotenv import load_dotenv

class PokemonDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.image_files = list(self.image_dir.glob('*.png'))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        return {"image": image}

def train():
    # Load environment variables
    load_dotenv()
    
    # Get token from environment variable
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    if not hf_token:
        raise ValueError("Please set HUGGING_FACE_TOKEN in .env file")
    
    # Use token for authentication
    HfFolder.save_token(hf_token)
    
    # Download dataset directly from hub
    local_dir = "pokemon_data"
    if not os.path.exists(local_dir):
        snapshot_download(
            repo_id="huggan/pokemon",
            repo_type="dataset",
            local_dir=local_dir,
            token=True
        )
    
    # Create dataset from downloaded files
    dataset = PokemonDataset(os.path.join(local_dir, "data"))
    
    # Initialize the pipeline before optimizer setup
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        use_auth_token=True
    )
    
    # Set up scheduler
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # Training configuration
    learning_rate = 1e-5
    num_epochs = 5
    batch_size = 1  # Small batch size for CPU
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        # Set models to training mode
        pipeline.unet.train()
        pipeline.vae.eval()  # VAE should stay in eval mode
        
        progress_bar = tqdm(dataset)
        
        for batch in progress_bar:
            image = prepare_image(batch['image'])
            
            # Generate noise
            noise = torch.randn_like(image)
            timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (1,))
            
            # Get model prediction
            noisy_image = pipeline.scheduler.add_noise(image, noise, timesteps)
            noise_pred = pipeline.unet(noisy_image, timesteps)['sample']
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Save checkpoint after each epoch
        pipeline.save_pretrained(f"checkpoints/epoch_{epoch+1}")

if __name__ == "__main__":
    train() 