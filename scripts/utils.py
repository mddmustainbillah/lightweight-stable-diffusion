import torch
from torchvision import transforms
from PIL import Image

def prepare_image(image):
    """Prepare image for training"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    if isinstance(image, Image.Image):
        image = transform(image)
    else:  # If image is from dataset
        image = transform(Image.fromarray(image))
    
    return image.unsqueeze(0) 