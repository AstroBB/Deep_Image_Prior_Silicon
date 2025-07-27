#!/usr/bin/env python
"""
Deep Image Prior - Main denoising module
Optimized for Apple Silicon (M1/M2/M3) with MPS backend
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

from models import get_net
from utils.common_utils import *


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def denoise_image(
    image_path: str,
    num_iter: int = 3000,
    learning_rate: float = 0.01,
    reg_noise_std: float = 1./30.,
    save_path: Optional[str] = None,
    show_progress: bool = True
) -> Dict:
    """
    Denoise an image using Deep Image Prior
    
    Args:
        image_path: Path to noisy image
        num_iter: Number of iterations (default: 3000)
        learning_rate: Learning rate for Adam optimizer
        reg_noise_std: Regularization noise standard deviation
        save_path: Path to save denoised image (optional)
        show_progress: Print progress during optimization
        
    Returns:
        Dictionary with results including PSNR improvement and timings
    """
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load image
    img_pil = Image.open(image_path)
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    
    # Convert to numpy and normalize
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # HWC to CHW
    
    # Convert to torch tensor
    img_torch = torch.from_numpy(img_np).unsqueeze(0).to(device)
    
    # Network parameters (from original paper)
    input_depth = 32
    pad = 'reflection'
    
    # Create network
    net = get_net(
        input_depth, 'skip', pad,
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        upsample_mode='bilinear'
    ).to(device)
    
    # Initialize input noise
    net_input = get_noise(input_depth, 'noise', (img_pil.size[1], img_pil.size[0])).to(device)
    net_input_saved = net_input.detach().clone()
    
    # Setup optimizer and loss
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    
    # For exponential smoothing
    exp_weight = 0.99
    out_avg = None
    
    # Tracking variables
    best_out = None
    best_psnr = 0
    best_iter = 0
    
    # For backtracking
    last_net = None
    psnr_last = 0
    
    # Timing
    start_time = time.time()
    
    # Optimization loop
    for i in range(num_iter):
        optimizer.zero_grad()
        
        # Add regularization noise
        if reg_noise_std > 0:
            net_input = net_input_saved + (torch.randn_like(net_input_saved) * reg_noise_std)
        
        # Forward pass
        out = net(net_input)
        
        # Exponential smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
        
        # Calculate loss
        loss = mse(out, img_torch)
        loss.backward()
        optimizer.step()
        
        # Calculate PSNR
        out_np = out_avg.detach().cpu().squeeze(0).numpy()
        out_np = np.clip(out_np, 0, 1)
        
        mse_val = np.mean((img_np - out_np) ** 2)
        psnr = 10 * np.log10(1.0 / mse_val) if mse_val > 0 else float('inf')
        
        # Backtracking mechanism
        if i % 100 == 0 and i > 0:
            if psnr - psnr_last < -5:
                if show_progress:
                    print(f'Iteration {i}: Backtracking (PSNR drop: {psnr - psnr_last:.2f})')
                if last_net is not None:
                    for new_param, net_param in zip(last_net, net.parameters()):
                        net_param.data.copy_(new_param.to(device))
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psnr_last = psnr
        
        # Track best result
        if psnr > best_psnr:
            best_psnr = psnr
            best_out = out_avg.clone()
            best_iter = i
        
        # Progress update
        if show_progress and i % 100 == 0:
            print(f"Iteration {i}/{num_iter} - Loss: {loss.item():.6f} - PSNR: {psnr:.2f} dB")
    
    # Calculate final metrics
    total_time = time.time() - start_time
    
    # Prepare final output
    final_out = best_out.detach().cpu().squeeze(0).numpy()
    final_out = np.clip(final_out, 0, 1)
    final_out_pil = Image.fromarray((final_out.transpose(1, 2, 0) * 255).astype(np.uint8))
    
    # Save if requested
    if save_path:
        final_out_pil.save(save_path)
        print(f"Denoised image saved to: {save_path}")
    
    # Calculate improvement (assuming input is noisy)
    input_psnr = 20.0  # Approximate for sigma=25/255 noise
    
    results = {
        'denoised_image': final_out_pil,
        'best_psnr': best_psnr,
        'best_iteration': best_iter,
        'psnr_improvement': best_psnr - input_psnr,
        'total_time': total_time,
        'time_per_iter': total_time / num_iter,
        'device': str(device)
    }
    
    print(f"\nDenoising complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB at iteration {best_iter}")
    print(f"Total time: {total_time:.1f} seconds")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python denoise.py <image_path> [output_path] [iterations]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "denoised_output.png"
    num_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    
    results = denoise_image(
        image_path,
        num_iter=num_iter,
        save_path=output_path
    )
    
    print(f"\nResults saved to: {output_path}")