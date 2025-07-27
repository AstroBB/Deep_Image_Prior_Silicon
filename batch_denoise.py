#!/usr/bin/env python
"""
Batch processing for Deep Image Prior denoising
Process multiple images efficiently
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import concurrent.futures
from tqdm import tqdm

from denoise import denoise_image


def process_folder(
    input_dir: str,
    output_dir: str,
    num_iter: int = 3000,
    file_extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
    device: str = "mps",
    max_workers: int = 1
) -> List[Dict]:
    """
    Process all images in a folder
    
    Args:
        input_dir: Directory containing noisy images
        output_dir: Directory to save denoised images
        num_iter: Number of iterations per image
        file_extensions: List of image file extensions to process
        device: Device to use (mps/cuda/cpu)
        max_workers: Number of parallel workers (1 recommended for GPU)
        
    Returns:
        List of result dictionaries for each processed image
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    results = []
    
    def process_single(img_path):
        """Process a single image"""
        output_path = Path(output_dir) / f"denoised_{img_path.name}"
        
        try:
            result = denoise_image(
                str(img_path),
                num_iter=num_iter,
                save_path=str(output_path),
                show_progress=False  # Disable per-image progress for batch
            )
            result['input_path'] = str(img_path)
            result['output_path'] = str(output_path)
            return result
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None
    
    # Process with progress bar
    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        if max_workers == 1:
            # Sequential processing (recommended for GPU)
            for img_path in image_files:
                result = process_single(img_path)
                if result:
                    results.append(result)
                pbar.update(1)
        else:
            # Parallel processing (for CPU)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single, img_path): img_path 
                          for img_path in image_files}
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
    
    # Summary
    print(f"\nProcessed {len(results)} images successfully")
    avg_improvement = sum(r['psnr_improvement'] for r in results) / len(results)
    avg_time = sum(r['total_time'] for r in results) / len(results)
    
    print(f"Average PSNR improvement: {avg_improvement:.2f} dB")
    print(f"Average processing time: {avg_time:.1f} seconds per image")
    
    return results


def create_comparison_grid(
    results: List[Dict],
    output_path: str = "comparison_grid.png",
    max_images: int = 6
):
    """
    Create a grid showing original vs denoised comparisons
    
    Args:
        results: List of result dictionaries from batch processing
        output_path: Path to save the comparison grid
        max_images: Maximum number of images to include
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    n_images = min(len(results), max_images)
    fig, axes = plt.subplots(n_images, 2, figsize=(10, 5 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results[:n_images]):
        # Load original
        original = Image.open(result['input_path'])
        if original.mode != 'RGB':
            original = original.convert('RGB')
        
        # Load denoised
        denoised = result['denoised_image']
        
        # Plot
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Noisy Input")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(denoised)
        axes[i, 1].set_title(f"Denoised (+{result['psnr_improvement']:.1f} dB)")
        axes[i, 1].axis('off')
    
    plt.suptitle("Deep Image Prior - Batch Denoising Results", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison grid saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch denoise images using Deep Image Prior")
    parser.add_argument("input_dir", help="Directory containing noisy images")
    parser.add_argument("output_dir", help="Directory to save denoised images")
    parser.add_argument("--iterations", type=int, default=3000, help="Number of iterations (default: 3000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    
    args = parser.parse_args()
    
    # Process folder
    results = process_folder(
        args.input_dir,
        args.output_dir,
        num_iter=args.iterations,
        max_workers=args.workers
    )
    
    # Create comparison grid
    if results:
        create_comparison_grid(results, os.path.join(args.output_dir, "comparison_grid.png"))