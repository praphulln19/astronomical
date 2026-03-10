"""Generate synthetic astronomical images for testing the pipeline."""
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

def add_stars(img, n_stars=50, seed=42):
    """Add point sources (stars) to the image."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    
    for _ in range(n_stars):
        # Random position
        y = rng.integers(10, h - 10)
        x = rng.integers(10, w - 10)
        
        # Random brightness
        brightness = rng.uniform(0.3, 1.0)
        
        # Add Gaussian PSF (Point Spread Function)
        sigma = rng.uniform(0.8, 2.0)
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if 0 <= y + dy < h and 0 <= x + dx < w:
                    dist = np.sqrt(dx**2 + dy**2)
                    intensity = brightness * np.exp(-(dist**2) / (2 * sigma**2))
                    if img.ndim == 3:
                        img[y + dy, x + dx] = np.clip(img[y + dy, x + dx] + intensity, 0, 1)
                    else:
                        img[y + dy, x + dx] = np.clip(img[y + dy, x + dx] + intensity, 0, 1)
    
    return img

def add_noise(img, noise_level=0.05, seed=42):
    """Add Gaussian and Poisson noise to simulate real astronomical data."""
    rng = np.random.default_rng(seed)
    
    # Poisson noise (photon counting noise)
    # Scale to reasonable counts, add noise, scale back
    scaled = img * 1000
    noisy = rng.poisson(scaled).astype(float) / 1000
    
    # Gaussian read noise
    gaussian_noise = rng.normal(0, noise_level, img.shape)
    noisy = noisy + gaussian_noise
    
    return np.clip(noisy, 0, 1)

def generate_image(width=1024, height=1024, n_stars=100, noise_level=0.05, seed=42):
    """Generate a synthetic astronomical image."""
    rng = np.random.default_rng(seed)
    
    # Background with slight gradient
    y_grid, x_grid = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing='ij')
    background = 0.05 + 0.02 * (y_grid + x_grid) / 2
    
    # Add some random background variations
    background += rng.normal(0, 0.01, (height, width))
    background = np.clip(background, 0, 1)
    
    # Create RGB image
    img = np.stack([background, background, background], axis=-1)
    
    # Add stars
    img = add_stars(img, n_stars=n_stars, seed=seed)
    
    # Add noise
    img = add_noise(img, noise_level=noise_level, seed=seed + 1000)
    
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic astronomical images')
    parser.add_argument('--output_dir', type=Path, default=Path('data/raw'), help='Output directory')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    parser.add_argument('--stars', type=int, default=100, help='Number of stars per image')
    parser.add_argument('--noise', type=float, default=0.05, help='Noise level')
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(args.num_images):
        img = generate_image(
            width=args.width,
            height=args.height,
            n_stars=args.stars,
            noise_level=args.noise,
            seed=42 + i
        )
        
        # Save as PNG
        img_pil = Image.fromarray(img, mode='RGB')
        output_path = args.output_dir / f'synthetic_astro_{i:03d}.png'
        img_pil.save(output_path)
        print(f'Generated: {output_path}')
    
    print(f'\nSuccessfully generated {args.num_images} synthetic images in {args.output_dir}')

if __name__ == '__main__':
    main()
