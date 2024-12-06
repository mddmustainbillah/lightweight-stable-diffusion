# Lightweight Stable Diffusion Fine-tuning

This project implements a lightweight version of Stable Diffusion fine-tuning for image-to-image generation, optimized to run on CPU. It allows you to fine-tune the model on custom datasets and generate new images based on input images and text prompts.

## Features

- Fine-tune Stable Diffusion model on custom datasets
- Image-to-image generation with text prompts
- CPU-friendly implementation
- Configurable training parameters
- Checkpoint saving and loading

## Prerequisites

- Python 3.10 or higher
- pip package manager
- At least 16GB RAM recommended
- Hugging Face account and access token

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/lightweight-stable-diffusion.git
cd lightweight-stable-diffusion
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Hugging Face token:

```bash
HUGGING_FACE_TOKEN=your_token_here
```

## Project Structure
```
├── scripts/
│   ├── train.py       # Training script
│   ├── test.py        # Testing/inference script
│   └── utils.py       # Utility functions
├── checkpoints/       # Model checkpoints (gitignored)
├── pokemon_data/      # Dataset directory (gitignored)
├── requirements.txt   # Project dependencies
├── .env              # Environment variables (gitignored)
└── README.md         # This file
```

## Usage

### Training

1. Prepare your dataset in the appropriate directory
2. Run the training script:

```bash
python scripts/train.py
```

The model checkpoints will be saved in the `checkpoints/` directory after each epoch.

### Testing

To generate images using your fine-tuned model:

```bash
python scripts/test.py
```

Modify the parameters in `test.py` to:
- Change the input image path
- Adjust the text prompt
- Select a different checkpoint
- Modify generation parameters

## Configuration

Key parameters in `train.py`:
- `learning_rate`: Training learning rate (default: 1e-5)
- `num_epochs`: Number of training epochs (default: 5)
- `batch_size`: Batch size for training (default: 1)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the model and dataset infrastructure
- [Stable Diffusion](https://stability.ai/stable-diffusion) by Stability AI

       