#!/usr/bin/env python
"""
Example usage of Deep Image Prior for image denoising
"""

from denoise import denoise_image
from add_noise import add_gaussian_noise
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def main():
    """
    Complete example: add noise to a clean image and then denoise it
    """
    
    # Step 1: Add noise to a clean image
    print("Step 1: Adding Gaussian noise to image...")
    clean_image_path = "sample_images/F16_GT.png"
    noisy_image_path, input_psnr = add_gaussian_noise(
        clean_image_path,
        sigma=25/255.,  # Standard deviation used in paper
        output_path="sample_images/F16_noisy.png"
    )
    
    # Step 2: Denoise using Deep Image Prior
    print("\nStep 2: Denoising with Deep Image Prior...")
    print("This will take approximately 3-4 minutes on Apple Silicon...")
    
    results = denoise_image(
        noisy_image_path,
        num_iter=3000,  # Recommended for 512x512 images
        save_path="sample_images/F16_denoised.png"
    )
    
    # Step 3: Display results
    print("\nStep 3: Creating comparison figure...")
    
    # Load images
    clean = Image.open(clean_image_path)
    noisy = Image.open(noisy_image_path)
    denoised = results['denoised_image']
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(clean)
    axes[0].set_title('Original Clean Image')
    axes[0].axis('off')
    
    axes[1].imshow(noisy)
    axes[1].set_title(f'Noisy (PSNR: {input_psnr:.2f} dB)')
    axes[1].axis('off')
    
    axes[2].imshow(denoised)
    axes[2].set_title(f'Denoised (PSNR: {results["best_psnr"]:.2f} dB)')
    axes[2].axis('off')
    
    plt.suptitle(f'Deep Image Prior - PSNR Improvement: +{results["psnr_improvement"]:.2f} dB', 
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('sample_images/example_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Device used: {results['device']}")
    print(f"Processing time: {results['total_time']:.1f} seconds")
    print(f"Time per iteration: {results['time_per_iter']*1000:.1f} ms")
    print(f"Best iteration: {results['best_iteration']}")
    print(f"PSNR improvement: +{results['psnr_improvement']:.2f} dB")
    print("\nNote: After ~3000-4000 iterations, the network may start overfitting")
    print("and memorizing the noise patterns instead of removing them.")


if __name__ == "__main__":
    main()