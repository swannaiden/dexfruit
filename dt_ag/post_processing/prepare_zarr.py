#!/usr/bin/env python3
"""
Zarr dataset filter and resize script.

This script copies data from an existing Zarr dataset and creates a new
filtered dataset containing only desired data.

"""

# ────────────────────────────────────────────────────────────────
#  Standard imports
# ────────────────────────────────────────────────────────────────
import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import zarr
from rich.console import Console
from tqdm import tqdm
import matplotlib.pyplot as plt
from rich.table import Table

# ────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────
dt_ag_root = Path(__file__).resolve().parents[1]

# Input Zarr dataset path
INPUT_ZARR_DIR = dt_ag_root / "data" / "2d_strawberry_dt" / "dt_blackberry_7-22_zarr"

# Output Zarr dataset path
OUTPUT_ZARR_DIR = dt_ag_root / "data" / "2d_strawberry_dt" / "dt_blackberry_7-22_zarr_for_cluster"

# Resize settings
ENABLE_RESIZE = True  # Set to False to keep original resolution
# TARGET_WIDTH = 220    # Target width for resizing
# TARGET_HEIGHT = 160   # Target height for resizing

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

# Crop settings
CROP_RS_FRONT = False  # Set to False to disable cropping
CROP_RS_SIDE = False
CROP_RS_WRIST = False
CROP_ZED = False

RS_FRONT_W_RANGE = (0, 640)
RS_FRONT_H_RANGE = (160, 320)

RS_SIDE_W_RANGE = (0, 640)
RS_SIDE_H_RANGE = (160, 320)

RS_WRIST_W_RANGE = (0, 640)
RS_WRIST_H_RANGE = (160, 320)

ZED_W_RANGE = (70, 320)
ZED_H_RANGE = (220, 340)

# Debug mode - set to True to process only first episode
DEBUGGING = False

CONSOLE = Console()

# ────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────
def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def apply_color_jitter(rgb_frames: np.ndarray, 
                      brightness: float = 0.0,
                      contrast: float = 0.0, 
                      saturation: float = 0.0,
                      hue: float = 0.0) -> np.ndarray:
    """
    Apply color jitter to RGB frames.
    
    Args:
        rgb_frames: Input frames in (T, C, H, W) or (T, H, W, C) format
        brightness: Brightness factor range (0.0 = no change)
        contrast: Contrast factor range (0.0 = no change)
        saturation: Saturation factor range (0.0 = no change)
        hue: Hue shift range (0.0 = no change)
    
    Returns:
        Color jittered frames in same format as input
    """
    if brightness == 0.0 and contrast == 0.0 and saturation == 0.0 and hue == 0.0:
        return rgb_frames
    
    # Convert to HWC format for processing
    if rgb_frames.ndim == 4 and rgb_frames.shape[-1] in (1, 3):
        # already (T, H, W, C)
        hwc = rgb_frames
        input_format = 'hwc'
    elif rgb_frames.ndim == 4 and rgb_frames.shape[1] in (1, 3):
        # currently (T, C, H, W) → transpose
        hwc = rgb_frames.transpose(0, 2, 3, 1)
        input_format = 'chw'
    else:
        raise ValueError(f"Expected (T,H,W,C) or (T,C,H,W), got {rgb_frames.shape}")
    
    T, H, W, C = hwc.shape
    jittered_hwc = np.zeros_like(hwc)
    
    for t in range(T):
        frame = hwc[t].astype(np.float32)
        
        # Apply brightness jitter
        if brightness > 0.0:
            brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
            frame = frame * brightness_factor
        
        # Apply contrast jitter
        if contrast > 0.0:
            contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
            mean = np.mean(frame)
            frame = (frame - mean) * contrast_factor + mean
        
        # Convert to HSV for saturation and hue adjustments
        if saturation > 0.0 or hue > 0.0:
            # Convert RGB to HSV
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Apply saturation jitter
            if saturation > 0.0:
                saturation_factor = 1.0 + np.random.uniform(-saturation, saturation)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
            
            # Apply hue jitter
            if hue > 0.0:
                hue_shift = np.random.uniform(-hue, hue) * 180  # Convert to OpenCV hue range
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Convert back to RGB
            hsv = np.clip(hsv, 0, [180, 255, 255]).astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Clip values to valid range
        jittered_hwc[t] = np.clip(frame, 0, 255).astype(hwc.dtype)
    
    # Convert back to original format
    if input_format == 'chw':
        return jittered_hwc.transpose(0, 3, 1, 2)
    else:
        return jittered_hwc

def resize_rgb_frames(rgb_frames: np.ndarray, target_size: Tuple[int,int]) -> np.ndarray:
    """
    Resize RGB frames to target size, returning (T, C, H, W).

    Accepts either:
      - HWC: (T, H, W, C)
      - CHW: (T, C, H, W)
    """
    target_w, target_h = target_size

    # Step 1: get everything into HWC-per-frame
    if rgb_frames.ndim == 4 and rgb_frames.shape[-1] in (1,3):
        # already (T, H, W, C)
        hwc = rgb_frames
    elif rgb_frames.ndim == 4 and rgb_frames.shape[1] in (1,3):
        # currently (T, C, H, W) → transpose
        hwc = rgb_frames.transpose(0,2,3,1)  # → (T, H, W, C)
    else:
        raise ValueError(f"Expected (T,H,W,C) or (T,C,H,W), got {rgb_frames.shape}")

    T, H, W, C = hwc.shape
    out_hwc = np.zeros((T, target_h, target_w, C), dtype=hwc.dtype)

    # Step 2: resize each frame in HWC format
    for t in range(T):
        out_hwc[t] = cv2.resize(hwc[t],
                                (target_w, target_h),
                                interpolation=cv2.INTER_LINEAR)

    # Step 3: convert back to CHW-per-frame
    return out_hwc.transpose(0, 3, 1, 2)    # → (T, C, H, W)

def crop_rgb_frames(rgb_frames: np.ndarray, 
                   y_range: Tuple[int,int],
                   x_range: Tuple[int,int]) -> np.ndarray:
    """
    Crop RGB frames to target size with optional offset, returning (T, C, H, W).
    
    Args:
        rgb_frames: Input frames
        y_range: (start, end) of crop in y-direction
        x_range: (start, end) of crop in x-direction

    Accepts either:
      - HWC: (T, H, W, C)
      - CHW: (T, C, H, W)
    """

    # Step 1: get everything into HWC-per-frame
    if rgb_frames.ndim == 4 and rgb_frames.shape[-1] in (1,3):
        # already (T, H, W, C)
        hwc = rgb_frames
    elif rgb_frames.ndim == 4 and rgb_frames.shape[1] in (1,3):
        # currently (T, C, H, W) → transpose
        hwc = rgb_frames.transpose(0,2,3,1)  # → (T, H, W, C)
    else:
        raise ValueError(f"Expected (T,H,W,C) or (T,C,H,W), got {rgb_frames.shape}")

    T, H, W, C = hwc.shape
    
    # Step 2: crop each frame
    cropped_hwc = hwc[:, x_range[0]:x_range[1], y_range[0]:y_range[1], :]
    
    # Step 3: convert back to CHW-per-frame
    return cropped_hwc.transpose(0, 3, 1, 2)    # → (T, C, H, W)

def get_episode_names(zarr_root) -> list:
    """Get sorted list of episode names from Zarr root."""
    episode_names = [key for key in zarr_root.keys() if key.startswith("episode_")]
    return sorted(episode_names)

def check_required_arrays(episode_group, desired_data_names: list, episode_name: str) -> bool:
    """Check if episode has all required arrays."""
    for array_name in desired_data_names:
        if array_name not in episode_group:
            CONSOLE.log(f"[yellow]Episode {episode_name} missing {array_name}, skipping")
            return False
    
    return True

def inspect_zarr_dataset(zarr_path: Path, episode_idx: int = 0, visualize: bool = True) -> None:
    """
    Inspect a single episode from the Zarr dataset to understand data shapes.
    
    Args:
        zarr_path: Path to the Zarr dataset
        episode_idx: Which episode to inspect (default: 0)
        visualize: Whether to create visualizations (default: True)
    """
    
    if not zarr_path.exists():
        CONSOLE.log(f"[red]Zarr path does not exist: {zarr_path}")
        return
    
    try:
        root = zarr.open(zarr_path, mode="r")
        
        # List all available episodes
        episodes = [key for key in root.keys() if key.startswith("episode_")]
        episodes.sort()
        
        if not episodes:
            CONSOLE.log("[red]No episodes found in Zarr dataset!")
            return
        
        CONSOLE.log(f"[green]Found {len(episodes)} episodes in dataset")
        CONSOLE.log(f"[blue]Episodes: {episodes[0]} to {episodes[-1]}")
        
        # Select episode to inspect
        if episode_idx >= len(episodes):
            CONSOLE.log(f"[yellow]Episode index {episode_idx} out of range, using episode 0")
            episode_idx = 0
        
        episode_name = episodes[episode_idx]
        episode_group = root[episode_name]
        
        CONSOLE.log(f"\n[bold blue]Inspecting {episode_name}:")
        
        # Create a table for the results
        table = Table(title=f"Data Shapes in {episode_name}")
        table.add_column("Array Name", style="cyan", no_wrap=True)
        table.add_column("Shape", style="magenta")
        table.add_column("Data Type", style="green")
        table.add_column("Min Value", style="yellow")
        table.add_column("Max Value", style="yellow")
        table.add_column("Notes", style="white")
        
        # Inspect each array in the episode
        for array_name in sorted(episode_group.keys()):
            array = episode_group[array_name]
            
            # Get basic info
            shape_str = str(array.shape)
            dtype_str = str(array.dtype)
            
            # Calculate min/max for small arrays or sample for large ones
            try:
                if array.size < 1000000:  # For reasonably sized arrays
                    min_val = np.min(array[:])
                    max_val = np.max(array[:])
                else:  # For very large arrays, sample
                    sample = array[:10] if len(array.shape) > 0 else array[:]
                    min_val = np.min(sample)
                    max_val = np.max(sample)
                min_str = f"{min_val:.3f}" if isinstance(min_val, (float, np.floating)) else str(min_val)
                max_str = f"{max_val:.3f}" if isinstance(max_val, (float, np.floating)) else str(max_val)
            except Exception as e:
                min_str = "N/A"
                max_str = "N/A"
            
            # Add notes based on array name and properties
            notes = ""
            if array_name in ['rs_side_rgb', 'rs_front_rgb', 'rs_color_images']:
                notes = "RGB images"
            elif array_name in ['rs_depth', 'zed_depth']:
                notes = "Depth images"
            elif array_name == 'zed_pcd':
                notes = "Point clouds (T, N_points, 6) - [x,y,z,r,g,b]"
            elif array_name in ['agent_pos', 'pose']:
                notes = "Robot poses (T, 7) - [x,y,z,qw,qx,qy,qz]"
            elif array_name == 'action':
                notes = "Actions (T, 7) - pose deltas"
            
            table.add_row(array_name, shape_str, dtype_str, min_str, max_str, notes)
        
        CONSOLE.print(table)
        
        # Print episode attributes
        if hasattr(episode_group, 'attrs') and episode_group.attrs:
            CONSOLE.log(f"\n[bold blue]Episode Attributes:")
            for attr_name, attr_value in episode_group.attrs.items():
                CONSOLE.log(f"  {attr_name}: {attr_value}")
        
        # Visualize first frames if requested
        if visualize:
            CONSOLE.log(f"\n[bold green]Creating visualizations...")
            visualize_first_frames(episode_group, episode_name)
        
    except Exception as e:
        CONSOLE.log(f"[red]Error inspecting Zarr dataset: {e}")
        import traceback
        traceback.print_exc()

def visualize_first_frames(episode_group, episode_name: str) -> None:
    """
    Visualize the first frame of each image array in the episode.
    
    Args:
        episode_group: Zarr episode group containing the arrays
        episode_name: Name of the episode for display purposes
    """
    # Find all image arrays
    image_arrays = []
    for array_name in episode_group.keys():
        array = episode_group[array_name]
        # Check if this looks like an image array (3D or 4D with reasonable dimensions)
        if len(array.shape) >= 3:
            # Common image array patterns
            if any(keyword in array_name.lower() for keyword in ['rgb', 'color', 'image', 'depth']):
                image_arrays.append(array_name)
            # Also check for arrays that have image-like dimensions
            elif len(array.shape) == 4 and array.shape[-1] in [1, 3, 4]:  # (T, H, W, C)
                image_arrays.append(array_name)
            elif len(array.shape) == 4 and array.shape[1] in [1, 3, 4]:   # (T, C, H, W)
                image_arrays.append(array_name)
            elif len(array.shape) == 3 and min(array.shape[1:]) > 50:     # (T, H, W) - likely depth
                image_arrays.append(array_name)
    
    if not image_arrays:
        CONSOLE.log("[yellow]No image arrays found to visualize")
        return
    
    CONSOLE.log(f"[green]Found {len(image_arrays)} image arrays to visualize: {image_arrays}")
    
    # Set up the plot
    n_images = len(image_arrays)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    if n_images == 1:
        axes = [axes]  # Make it iterable
    
    fig.suptitle(f'First Frame Visualization - {episode_name}', fontsize=16)
    
    for idx, array_name in enumerate(image_arrays):
        array = episode_group[array_name]
        ax = axes[idx]
        
        try:
            # Get the first frame
            first_frame = array[0]  # Shape: (H, W, C) or (C, H, W) or (H, W)
            
            # Handle different array formats
            if len(first_frame.shape) == 3:
                if first_frame.shape[-1] in [1, 3, 4]:  # (H, W, C) format
                    display_image = first_frame
                    if first_frame.shape[-1] == 1:  # Grayscale
                        display_image = first_frame.squeeze(-1)
                elif first_frame.shape[0] in [1, 3, 4]:  # (C, H, W) format
                    if first_frame.shape[0] == 1:  # Grayscale
                        display_image = first_frame[0]
                    else:  # RGB
                        display_image = first_frame.transpose(1, 2, 0)  # Convert to (H, W, C)
                else:
                    # Assume it's some other 3D format, take first slice
                    display_image = first_frame[:, :, 0]
            elif len(first_frame.shape) == 2:  # (H, W) - likely depth or grayscale
                display_image = first_frame
            else:
                CONSOLE.log(f"[yellow]Cannot visualize {array_name} with shape {first_frame.shape}")
                continue
            
            # Handle different data types and ranges
            if 'depth' in array_name.lower():
                # For depth images, use a colormap and handle potential inf/nan values
                display_image = np.nan_to_num(display_image, nan=0, posinf=0, neginf=0)
                im = ax.imshow(display_image, cmap='viridis')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # For RGB/color images
                if display_image.dtype == np.uint8:
                    # Already in 0-255 range
                    im = ax.imshow(display_image)
                elif display_image.max() <= 1.0:
                    # Assuming 0-1 range
                    im = ax.imshow(display_image)
                else:
                    # Scale to 0-255 if needed
                    display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
                    im = ax.imshow(display_image)
            
            # Set title and info
            shape_str = f"{array.shape}"
            dtype_str = f"{array.dtype}"
            ax.set_title(f'{array_name}\nShape: {shape_str}\nDType: {dtype_str}', fontsize=10)
            ax.axis('off')
            
            # Add value range info
            if len(first_frame.shape) <= 3:
                min_val = np.min(first_frame)
                max_val = np.max(first_frame)
                mean_val = np.mean(first_frame)
                ax.text(0.02, 0.98, f'Range: [{min_val:.2f}, {max_val:.2f}]\nMean: {mean_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
            
        except Exception as e:
            CONSOLE.log(f"[red]Error visualizing {array_name}: {e}")
            ax.text(0.5, 0.5, f'Error visualizing\n{array_name}\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(array_name)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path("first_frame_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    CONSOLE.log(f"[green]Visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()

# ────────────────────────────────────────────────────────────────
#  Main conversion routine
# ────────────────────────────────────────────────────────────────
def main() -> None:
    """Main function to filter and resize Zarr dataset."""
    
    # Check if input directory exists
    if not INPUT_ZARR_DIR.exists():
        CONSOLE.log(f"[red]Input Zarr directory does not exist: {INPUT_ZARR_DIR}")
        return
    
    # Create output directory
    ensure_dir(OUTPUT_ZARR_DIR)
    
    try:
        # Open input Zarr dataset
        input_root = zarr.open(INPUT_ZARR_DIR, mode="r")
        CONSOLE.log(f"[green]Opened input Zarr dataset: {INPUT_ZARR_DIR}")
    except Exception as e:
        CONSOLE.log(f"[red]Error opening input Zarr dataset: {e}")
        return
    
    # Get episode names
    episode_names = get_episode_names(input_root)
    if not episode_names:
        CONSOLE.log("[red]No episodes found in input dataset")
        return
    
    if DEBUGGING:
        episode_names = episode_names[:1]
        CONSOLE.log(f"[yellow]Debug mode: processing only {episode_names[0]}")
    
    CONSOLE.log(f"[blue]Found {len(episode_names)} episodes to process")
    
    # Open output Zarr dataset
    output_root = zarr.open(OUTPUT_ZARR_DIR, mode="w")
    
    # Process each episode
    successful_episodes = 0
    
    for episode_name in tqdm(episode_names, desc="Processing episodes"):
        try:
            CONSOLE.log(f"[blue]Processing {episode_name}")
            
            # Create an array of names of data that we want for this ablation and load data
            # Core data that must be present
            required_data_names = ['rs_front_rgb', 'rs_wrist_rgb', 'pose', 'action']
            # Optional data for backwards compatibility
            optional_data_names = ['rs_side_rgb', 'dt_left_diff', 'dt_right_diff']
            
            input_episode = input_root[episode_name]
            
            # Check if episode has all required arrays
            if not check_required_arrays(input_episode, required_data_names, episode_name):
                continue
            
            # Load required data
            desired_data = {name: input_episode[name][:] for name in required_data_names}
            
            # Load optional DT data if available
            for name in optional_data_names:
                if name in input_episode:
                    desired_data[name] = input_episode[name][:]
                    CONSOLE.log(f"[green]Found optional data: {name}")
                else:
                    CONSOLE.log(f"[yellow]Optional data not found: {name}")
            
            T = desired_data['pose'].shape[0]

            # Apply cropping if enabled
            if CROP_RS_FRONT and 'rs_front_rgb' in desired_data:
                CONSOLE.log(f"[blue]Applying center crop to RS frames from {desired_data['rs_front_rgb'].shape[1:3]} to ({RS_FRONT_W_RANGE}, {RS_FRONT_H_RANGE})")
                desired_data['rs_front_rgb'] = crop_rgb_frames(desired_data['rs_front_rgb'], RS_FRONT_W_RANGE, RS_FRONT_H_RANGE)

            if CROP_RS_SIDE and 'rs_side_rgb' in desired_data:
                CONSOLE.log(f"[blue]Applying center crop to RS frames from {desired_data['rs_side_rgb'].shape[1:3]} to ({RS_SIDE_W_RANGE}, {RS_SIDE_H_RANGE})")
                desired_data['rs_side_rgb'] = crop_rgb_frames(desired_data['rs_side_rgb'], RS_SIDE_W_RANGE, RS_SIDE_H_RANGE)

            if CROP_RS_WRIST and 'rs_wrist_rgb' in desired_data:
                CONSOLE.log(f"[blue]Applying center crop to RS frames from {desired_data['rs_wrist_rgb'].shape[1:3]} to ({RS_WRIST_W_RANGE}, {RS_WRIST_H_RANGE})")
                desired_data['rs_wrist_rgb'] = crop_rgb_frames(desired_data['rs_wrist_rgb'], RS_WRIST_W_RANGE, RS_WRIST_H_RANGE)

            if CROP_ZED and 'zed_rgb' in desired_data:
                CONSOLE.log(f"[blue]Applying center crop to ZED frames from {desired_data['zed_rgb'].shape[1:3]} to ({ZED_W_RANGE}, {ZED_H_RANGE})")
                desired_data['zed_rgb'] = crop_rgb_frames(desired_data['zed_rgb'], ZED_W_RANGE, ZED_H_RANGE)

            # Resize RGB frames if enabled
            if ENABLE_RESIZE:
                CONSOLE.log(f"[blue]Resizing RGB frames to ({TARGET_WIDTH}, {TARGET_HEIGHT})")
                if 'rs_side_rgb' in desired_data:
                    CONSOLE.log(f"[blue]Resizing rs_side_rgb from {desired_data['rs_side_rgb'].shape[2:4]} to ({TARGET_WIDTH}, {TARGET_HEIGHT})")
                    desired_data['rs_side_rgb'] = resize_rgb_frames(desired_data['rs_side_rgb'], (TARGET_WIDTH, TARGET_HEIGHT))
                if 'rs_front_rgb' in desired_data:
                    CONSOLE.log(f"[blue]Resizing rs_front_rgb from {desired_data['rs_front_rgb'].shape[2:4]} to ({TARGET_WIDTH}, {TARGET_HEIGHT})")
                    desired_data['rs_front_rgb'] = resize_rgb_frames(desired_data['rs_front_rgb'], (TARGET_WIDTH, TARGET_HEIGHT))
                if 'rs_wrist_rgb' in desired_data:
                    CONSOLE.log(f"[blue]Resizing rs_wrist_rgb from {desired_data['rs_wrist_rgb'].shape[2:4]} to ({TARGET_WIDTH}, {TARGET_HEIGHT})")
                    desired_data['rs_wrist_rgb'] = resize_rgb_frames(desired_data['rs_wrist_rgb'], (TARGET_WIDTH, TARGET_HEIGHT))
                if 'zed_rgb' in desired_data:
                    CONSOLE.log(f"[blue]Resizing zed_rgb from {desired_data['zed_rgb'].shape[2:4]} to ({TARGET_WIDTH}, {TARGET_HEIGHT})")
                    desired_data['zed_rgb'] = resize_rgb_frames(desired_data['zed_rgb'], (TARGET_WIDTH, TARGET_HEIGHT))
                # Resize DT diff data if present
                if 'dt_left_diff' in desired_data:
                    CONSOLE.log(f"[blue]Resizing dt_left_diff from {desired_data['dt_left_diff'].shape[2:4]} to ({TARGET_WIDTH}, {TARGET_HEIGHT})")
                    desired_data['dt_left_diff'] = resize_rgb_frames(desired_data['dt_left_diff'], (TARGET_WIDTH, TARGET_HEIGHT))
                if 'dt_right_diff' in desired_data:
                    CONSOLE.log(f"[blue]Resizing dt_right_diff from {desired_data['dt_right_diff'].shape[2:4]} to ({TARGET_WIDTH}, {TARGET_HEIGHT})")
                    desired_data['dt_right_diff'] = resize_rgb_frames(desired_data['dt_right_diff'], (TARGET_WIDTH, TARGET_HEIGHT))

            # Create output episode group
            output_episode = output_root.create_group(episode_name)
            for key, arr in desired_data.items():
                dtype   = np.uint8 if arr.dtype == np.uint8 else arr.dtype
                chunks  = (1, *arr.shape[1:]) if arr.ndim >= 3 else None
                output_episode.array(key, arr, dtype=dtype, chunks=chunks)
            
            # Copy episode attributes
            if hasattr(input_episode, 'attrs'):
                for attr_name, attr_value in input_episode.attrs.items():
                    output_episode.attrs[attr_name] = attr_value
            
            output_episode.attrs['length'] = T
            successful_episodes += 1
            CONSOLE.log(f"[green]✓ Completed {episode_name}")
            
        except Exception as e:
            CONSOLE.log(f"[red]Error processing {episode_name}: {e}")
            if episode_name in output_root:
                del output_root[episode_name]
            continue

    # ────────────────────────────────────────────────────────────
    #  Store one-shot preprocessing metadata at the Zarr **root**
    # ────────────────────────────────────────────────────────────
    # Update to include all possible data sources in metadata
    all_possible_data_names = required_data_names + optional_data_names
    
    preprocess_meta = {
        "resize": {
            "enabled": ENABLE_RESIZE,
            "target": [TARGET_WIDTH, TARGET_HEIGHT],          # [W, H]
        },
        "crop": {
            "rs_side": {
                "present": "rs_side_rgb" in all_possible_data_names,
                "enabled": CROP_RS_FRONT,
                "crop_range":   [RS_FRONT_W_RANGE, RS_FRONT_H_RANGE],    # [W, H]
            },
            "rs_front": {
                "present": "rs_front_rgb" in all_possible_data_names,
                "enabled": CROP_RS_FRONT,
                "crop_range":   [RS_FRONT_W_RANGE, RS_FRONT_H_RANGE],
            },
            "rs_wrist": {
                "present": "rs_wrist_rgb" in all_possible_data_names,
                "enabled": CROP_RS_WRIST,
                "crop_range":   [RS_WRIST_W_RANGE, RS_WRIST_H_RANGE],
            },
            "zed": {
                "present": "zed_rgb" in all_possible_data_names,
                "enabled": CROP_ZED,
                "crop_range":   [ZED_W_RANGE, ZED_H_RANGE],
            },
        },
        "color_jitter": {
            "enabled":    ENABLE_COLOR_JITTER,
            "brightness": COLOR_JITTER_BRIGHTNESS,
            "contrast":   COLOR_JITTER_CONTRAST,
            "saturation": COLOR_JITTER_SATURATION,
            "hue":        COLOR_JITTER_HUE,
        },
        "dt_data": {
            "dt_left_diff_present": any('dt_left_diff' in output_root[ep] for ep in output_root.keys() if ep.startswith('episode_')),
            "dt_right_diff_present": any('dt_right_diff' in output_root[ep] for ep in output_root.keys() if ep.startswith('episode_')),
        },
    }
    output_root.attrs["preprocess"] = preprocess_meta
    # ────────────────────────────────────────────────────────────

    # Final summary
    CONSOLE.log(f"[green]Processing complete!")
    CONSOLE.log(f"[green]Successfully processed: {successful_episodes}/{len(episode_names)} episodes")
    CONSOLE.log(f"[green]Output dataset saved to: {OUTPUT_ZARR_DIR}")

    # Transformation log (unchanged)
    transformations = []
    if CROP_RS_FRONT:
        transformations.append(f"RS cropped ({RS_FRONT_W_RANGE}, {RS_FRONT_H_RANGE})")
    if CROP_RS_SIDE:
        transformations.append(f"RS side cropped ({RS_SIDE_W_RANGE}, {RS_SIDE_H_RANGE})")
    if CROP_RS_WRIST:
        transformations.append(f"RS wrist cropped ({RS_WRIST_W_RANGE}, {RS_WRIST_H_RANGE})")
    if CROP_ZED:
        transformations.append(f"ZED cropped ({ZED_W_RANGE}, {ZED_H_RANGE})")
    if ENABLE_COLOR_JITTER:
        transformations.append(f"Color jitter applied (brightness={COLOR_JITTER_BRIGHTNESS}, contrast={COLOR_JITTER_CONTRAST})")
    if ENABLE_RESIZE:
        transformations.append(f"Resized to: {TARGET_HEIGHT}x{TARGET_WIDTH}")
    
    if transformations:
        CONSOLE.log(f"[green]Transformations applied: {', '.join(transformations)}")
    else:
        CONSOLE.log(f"[green]RGB frames kept at original resolution")
    
    if successful_episodes > 0:
        sample_episode = output_root[episode_names[0]]
        CONSOLE.log(f"[cyan]Dataset info:")
        CONSOLE.log(f"[cyan]  - Episodes: {successful_episodes}")
        CONSOLE.log(f"[cyan]  - RS side RGB shape: {sample_episode['rs_side_rgb'].shape}") if "rs_side_rgb" in sample_episode else None
        CONSOLE.log(f"[cyan]  - RS front RGB shape: {sample_episode['rs_front_rgb'].shape}") if "rs_front_rgb" in sample_episode else None
        CONSOLE.log(f"[cyan]  - RS wrist RGB shape: {sample_episode['rs_wrist_rgb'].shape}") if "rs_wrist_rgb" in sample_episode else None
        CONSOLE.log(f"[cyan]  - ZED RGB shape: {sample_episode['zed_rgb'].shape}") if "zed_rgb" in sample_episode else None
        CONSOLE.log(f"[cyan]  - DT left diff shape: {sample_episode['dt_left_diff'].shape}") if "dt_left_diff" in sample_episode else CONSOLE.log(f"[cyan]  - DT left diff: Not present")
        CONSOLE.log(f"[cyan]  - DT right diff shape: {sample_episode['dt_right_diff'].shape}") if "dt_right_diff" in sample_episode else CONSOLE.log(f"[cyan]  - DT right diff: Not present")
        CONSOLE.log(f"[cyan]  - Pose shape: {sample_episode['pose'].shape}")
        CONSOLE.log(f"[cyan]  - Action shape: {sample_episode['action'].shape}")

    # Whether to create visualizations (set to False to skip)
    VISUALIZE = True
    
    CONSOLE.log(f"[bold blue]Inspecting Zarr Dataset")
    CONSOLE.log(f"Path: {OUTPUT_ZARR_DIR}")
    CONSOLE.log(f"Episode: {0}")
    CONSOLE.log(f"Visualize: {VISUALIZE}")
    
    inspect_zarr_dataset(OUTPUT_ZARR_DIR, 0, VISUALIZE)

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()