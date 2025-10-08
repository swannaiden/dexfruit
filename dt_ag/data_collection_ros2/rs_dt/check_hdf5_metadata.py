#!/usr/bin/env python3

import os
import h5py
import numpy as np

def check_hdf5_structure():
    """Check the data structure of the first episode HDF5 file."""
    
    # Hardcoded path to the first episode
    data_path = os.path.join(os.getcwd(), "demo_data", "episode_0.hdf5")
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return
    
    print(f"Analyzing HDF5 file: {data_path}")
    print("=" * 60)
    
    try:
        with h5py.File(data_path, "r") as f:
            print(f"File size: {os.path.getsize(data_path) / (1024*1024):.2f} MB")
            print(f"Number of datasets: {len(f.keys())}")
            print()
            
            # Iterate through all datasets
            for dataset_name in f.keys():
                dataset = f[dataset_name]
                
                print(f"Dataset: '{dataset_name}'")
                print(f"  Shape: {dataset.shape}")
                print(f"  Data type: {dataset.dtype}")
                print(f"  Size in memory: {dataset.nbytes / (1024*1024):.2f} MB")
                
                # Additional info based on dataset type
                if len(dataset.shape) == 1:
                    print(f"  Length: {dataset.shape[0]} samples")
                elif len(dataset.shape) == 2:
                    print(f"  Dimensions: {dataset.shape[0]} samples x {dataset.shape[1]} features")
                elif len(dataset.shape) == 4:
                    print(f"  Image stack: {dataset.shape[0]} images, {dataset.shape[2]}x{dataset.shape[3]} pixels, {dataset.shape[1]} channels")
                elif len(dataset.shape) == 3:
                    dims = "x".join(map(str, dataset.shape[1:]))
                    print(f"  3-D stack: {dataset.shape[0]} samples, each {dims}")
                
                # Show compression info if available
                if dataset.compression:
                    print(f"  Compression: {dataset.compression}")
                
                # Show sample values for small datasets
                if dataset.size <= 50:
                    print(f"  Sample values: {dataset[...].flatten()[:10]}")
                elif len(dataset.shape) <= 2:
                    print(f"  First few values: {dataset[...].flatten()[:5]}")
                    if dataset.shape[0] > 1:
                        print(f"  Last few values: {dataset[...].flatten()[-5:]}")
                
                print()
            
            # Summary statistics
            total_samples = None
            image_datasets = []
            scalar_datasets = []
            
            for dataset_name in f.keys():
                dataset = f[dataset_name]
                if len(dataset.shape) >= 3:  # Image data
                    image_datasets.append(dataset_name)
                    if total_samples is None:
                        total_samples = dataset.shape[0]
                elif len(dataset.shape) <= 2:  # Scalar/vector data
                    scalar_datasets.append(dataset_name)
                    if total_samples is None:
                        total_samples = dataset.shape[0]
            
            print("SUMMARY:")
            print(f"  Total samples/frames: {total_samples}")
            print(f"  Scalar datasets: {scalar_datasets}")
            print(f"  Image datasets: {image_datasets}")
            
            # Check data consistency
            print("\nDATA CONSISTENCY CHECK:")
            lengths = {}
            for dataset_name in f.keys():
                dataset = f[dataset_name]
                lengths[dataset_name] = dataset.shape[0]
            
            unique_lengths = set(lengths.values())
            if len(unique_lengths) == 1:
                print("  ✓ All datasets have consistent length")
            else:
                print("  ✗ Inconsistent dataset lengths:")
                for name, length in lengths.items():
                    print(f"    {name}: {length}")
    
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")

if __name__ == "__main__":
    check_hdf5_structure()