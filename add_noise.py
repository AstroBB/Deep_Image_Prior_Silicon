#!/usr/bin/env python
"""
Add synthetic noise to images for testing Deep Image Prior
"""

import numpy as np
from PIL import Image
import os
from typing import Optional, Tuple


def add_gaussian_noise(
    image_path: str,
    sigma: float = 25/255.,
    output_path: Optional[str] = None
) -> Tuple[str, float]:
    """
    Add Gaussian noise to an image
    
    Args:
        image_path: Path to clean image
        sigma: Noise standard deviation (default: 25/255 as in paper)
        output_path: Path to save noisy image (optional)
        
    Returns:
        Tuple of (output_path, psnr)
    """
    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array and normalize to [0,1]
    img_np = np.array(img).astype(np.float32) / 255.
    
    # Add Gaussian noise
    noise = np.random.normal(0, sigma, img_np.shape)
    img_noisy_np = img_np + noise
    
    # Clip to valid range
    img_noisy_np = np.clip(img_noisy_np, 0, 1)
    
    # Convert back to uint8
    img_noisy = (img_noisy_np * 255).astype(np.uint8)
    
    # Save
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_noisy{ext}"
    
    Image.fromarray(img_noisy).save(output_path)
    
    # Calculate PSNR
    mse = np.mean((img_np - img_noisy_np) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    print(f"Added noise with sigma={sigma:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Saved to: {output_path}")
    
    return output_path, psnr


def add_noise_to_folder(
    input_dir: str,
    output_dir: str,
    sigma: float = 25/255.
):
    """
    Add noise to all images in a folder
    
    Args:
        input_dir: Directory containing clean images
        output_dir: Directory to save noisy images
        sigma: Noise standard deviation
    """
    import pathlib
    
    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process all images
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(pathlib.Path(input_dir).glob(f"*{ext}"))
        image_files.extend(pathlib.Path(input_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    for img_path in image_files:
        output_path = pathlib.Path(output_dir) / f"{img_path.stem}_noisy{img_path.suffix}"
        
        try:
            _, psnr = add_gaussian_noise(str(img_path), sigma, str(output_path))
            results.append((img_path.name, psnr))
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Summary
    if results:
        avg_psnr = sum(r[1] for r in results) / len(results)
        print(f"\nProcessed {len(results)} images")
        print(f"Average PSNR: {avg_psnr:.2f} dB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add Gaussian noise to images")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-o", "--output", help="Output path (optional)")
    parser.add_argument("-s", "--sigma", type=float, default=25/255., 
                       help="Noise standard deviation (default: 25/255)")
    parser.add_argument("--batch", action="store_true", 
                       help="Process entire directory")
    
    args = parser.parse_args()
    
    if args.batch:
        # Process directory
        if not args.output:
            args.output = args.input + "_noisy"
        add_noise_to_folder(args.input, args.output, args.sigma)
    else:
        # Process single image
        add_gaussian_noise(args.input, args.sigma, args.output)