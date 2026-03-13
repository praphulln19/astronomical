"""
Standalone script to use the trained astronomical denoising model.
Share this file along with:
1. The checkpoint file (e.g., ckpt_5805.pt)
2. The unet_blindspot.py file (keep it in the same directory or in models/ subdirectory)

Usage:
    python use_checkpoint_standalone.py --checkpoint path/to/ckpt_5805.pt --input path/to/image.png --output denoised.png
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
from pathlib import Path

# =============================
# Model Architecture (UNetBlindspot)
# =============================
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )

class UNetBlindspot(nn.Module):
    def __init__(self, in_ch=3, base=48):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base*2)
        self.enc3 = conv_block(base*2, base*4)
        self.enc4 = conv_block(base*4, base*8)

        self.pool = nn.MaxPool2d(2)

        self.bott = conv_block(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = conv_block(base*16, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)

        self.head = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bott(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


# =============================
# Load Model from Checkpoint
# =============================
def load_model(checkpoint_path, device="cpu"):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model with same architecture as training
    model = UNetBlindspot(in_ch=3, base=48)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict (checkpoint may contain optimizer state too)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model


# =============================
# Image Processing Functions
# =============================
def load_and_preprocess_image(image_path, device="cpu"):
    """Load image and convert to tensor with proper padding."""
    # Load image
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Convert to tensor [1, 3, H, W]
    img_array = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    
    # Pad to multiple of 16 (required for UNet)
    _, _, h, w = tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    
    return tensor, (h, w)


def postprocess_and_save(tensor, original_size, output_path):
    """Convert tensor back to image and save."""
    h, w = original_size
    
    # Remove padding and convert to numpy
    tensor = tensor[:, :, :h, :w]
    img_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Clip and convert to uint8
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    
    # Save
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"✅ Denoised image saved to {output_path}")


# =============================
# Main Inference Function
# =============================
def denoise_image(checkpoint_path, input_path, output_path):
    """Denoise an image using the trained model."""
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Load and preprocess image
    print(f"Loading image from {input_path}...")
    tensor, original_size = load_and_preprocess_image(input_path, device)
    
    # Run inference
    print("Running denoising...")
    with torch.no_grad():
        output = model(tensor)
        output = torch.clamp(output, 0, 1)
    
    # Save result
    postprocess_and_save(output, original_size, output_path)
    print("Done!")


# =============================
# Command Line Interface
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise astronomical images using trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (e.g., ckpt_5805.pt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input noisy image")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save denoised image")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        exit(1)
    
    if not Path(args.input).exists():
        print(f"❌ Error: Input image not found: {args.input}")
        exit(1)
    
    # Run denoising
    denoise_image(args.checkpoint, args.input, args.output)
