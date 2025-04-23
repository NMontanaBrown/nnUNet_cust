#!/usr/bin/env python

"""
Ultra-Simple nnUNet Inference CLI

A minimalist command-line tool for running nnUNet inference on NIfTI (.nii.gz) files
with direct model loading and explicit device selection (CUDA, CPU, or MPS/Metal).

Usage:
    python simple_inference.py --device mps --input /path/to/input.nii.gz --model /path/to/model.model --plans /path/to/plans.pkl

Options:
    --device: Specify the device to use (cuda, cpu, or mps)
    --input: Path to input .nii.gz file
    --model: Path to the model file (.model)
    --plans: Path to the plans.pkl file
    --output: Path to output segmentation file (optional)
    --no-tta: Disable test-time augmentation (faster but potentially less accurate)
    --no-mixed-precision: Disable mixed precision inference
    --fast: Use fast inference mode (uses less memory, potentially faster)
    --fast-single-pass: Use fully convolutional single-pass inference (fastest, but requires entire image to fit in memory)
"""

import os
import tqdm
import argparse
import time
import pickle
import numpy as np
import torch
import SimpleITK as sitk
from typing import Union, List, Tuple, Dict

# Minimum required imports from nnUNet
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.network_architecture.generic_UNet import Generic_UNet


def load_and_preprocess(input_path: str, plans: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Basic preprocessing function for nnUNet inference
    
    Args:
        input_path: Path to input .nii.gz file
        plans: The loaded plans dictionary containing preprocessing info
        
    Returns:
        Tuple of (preprocessed_data, properties_dict)
    """
    # Load image
    image = sitk.ReadImage(input_path)
    spacing = np.array(image.GetSpacing())[::-1]  # sitk to numpy order
    image_data = sitk.GetArrayFromImage(image).astype(np.float32)
    
    # Add channel dimension if needed
    if len(image_data.shape) == 3:
        image_data = image_data[None]
    
    # Basic preprocessing
    # 1. Transpose according to plans if needed
    if "transpose_forward" in plans:
        transpose_forward = plans["transpose_forward"]
        image_data = image_data.transpose([0] + [i + 1 for i in transpose_forward])
    
    # 2. Simple Z-score normalization (basic)
    mean = np.mean(image_data)
    std = np.std(image_data)
    image_data = (image_data - mean) / (std + 1e-8)
    
    # Create properties dict for saving later
    properties = {
        "original_size_of_raw_data": image_data.shape[1:],
        "original_spacing": spacing,
        "spacing": spacing,
        "size_after_cropping": image_data.shape[1:],
        "size_after_resampling": image_data.shape[1:],
        "itk_origin": image.GetOrigin(),
        "itk_spacing": image.GetSpacing(),
        "itk_direction": image.GetDirection()
    }
    
    return image_data, properties


def load_network(model_file: str, plans_file: str, device: str, initialize_with_checkpoint: bool = True) -> Tuple[torch.nn.Module, Dict]:
    """
    Load nnUNet model and plans directly without using trainer classes
    
    Args:
        model_file: Path to the model file (.model)
        plans_file: Path to the plans.pkl file
        device: Device to use ('cuda', 'cpu', or 'mps')
        
    Returns:
        Tuple of (loaded_network, plans_dict)
    """
    # Load plans file
    with open(plans_file, 'rb') as f:
        plans = pickle.load(f)
        
    # Debug plans structure
    print(f"Plans keys: {list(plans.keys())}")
    
    # Check if using different plans structure (TotalSegmentator might use a different structure)
    if "plans_per_stage" in plans and isinstance(plans["plans_per_stage"], list) and len(plans["plans_per_stage"]) > 0:
        print("Using plans_per_stage structure")
        # Use the first stage's plans
        stage_plans = plans["plans_per_stage"][0]
        print(f"Stage plans keys: {list(stage_plans.keys())}")
        
        # Update the main plans with stage-specific details
        for key in ["patch_size", "pool_op_kernel_sizes", "conv_kernel_sizes", "num_pool_per_axis"]:
            if key in stage_plans:
                plans[key] = stage_plans[key]
    
    # Load model file with explicit weights_only=False to handle PyTorch 2.6+ security changes
    try:
        # First try with weights_only=False to handle pickle security changes in PyTorch 2.6+
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load model: {e}")
        # Try to add numpy scalar as a safe global if necessary
        try:
            import numpy as np
            from torch.serialization import safe_globals
            with safe_globals([np.core.multiarray.scalar]):
                checkpoint = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
        except:
            print("Could not load model using torch.load. Please check that your model file is compatible with your PyTorch version.")
            raise
    
    # Instead of trying to reconstruct network parameters, load the architecture directly
    # This ensures we have the exact same architecture that was used for training
    model_pkl_path = os.path.join(os.path.dirname(model_file), "model_final_checkpoint.model.pkl")
    
    if os.path.exists(model_pkl_path):
        print(f"Loading network architecture from {model_pkl_path}")
        with open(model_pkl_path, 'rb') as f:
            model_dict = pickle.load(f)
            
        # Get trainer class from the pickle file
        trainer_class_name = model_dict['class']
        print(f"Original trainer class: {trainer_class_name}")
        
        # Use original plans from the pickle
        orig_plans = model_dict['plans']
        
        # Extract correct network parameters
        num_input_channels = orig_plans.get("num_modalities", 1)
        base_num_features = orig_plans.get("base_num_features", 32)
        num_classes = orig_plans.get("num_classes", 18) + 1  # background + foreground classes
        
        # Fix max_num_features - critical for matching the model dimensions
        # For 3D nnUNet, default is 320 for 3D networks
        max_num_features = 320  # This is the standard for 3D nnUNet
        
        # Get the pool/conv parameters from plans_per_stage
        stage0_plans = orig_plans["plans_per_stage"][0]
        net_pool_per_axis = stage0_plans["pool_op_kernel_sizes"]
        conv_kernel_sizes = stage0_plans["conv_kernel_sizes"]
        
        # Other important parameters
        num_conv_per_stage = orig_plans.get("conv_per_stage", 2)
        deep_supervision = True  # Almost always True for nnUNet
    else:
        print(f"Warning: Couldn't find {model_pkl_path}, using parameters from plans.pkl")
        # Extract from regular plans as fallback
        num_input_channels = plans.get("num_modalities", 1)
        base_num_features = plans.get("base_num_features", 32)
        num_classes = plans.get("num_classes", 18) + 1
        max_num_features = 320  # Standard for 3D nnUNet
        
        # Get from plans_per_stage
        if "plans_per_stage" in plans and len(plans["plans_per_stage"]) > 0:
            stage0_plans = plans["plans_per_stage"][0]
            net_pool_per_axis = stage0_plans.get("pool_op_kernel_sizes", [[2, 2, 2]] * 5)
            conv_kernel_sizes = stage0_plans.get("conv_kernel_sizes", [[3, 3, 3]] * 6)
        else:
            net_pool_per_axis = [[2, 2, 2]] * 5
            conv_kernel_sizes = [[3, 3, 3]] * 6
        
        num_conv_per_stage = plans.get("conv_per_stage", 2)
        deep_supervision = True
    
    print(f"Network parameters:")
    print(f"- input_channels: {num_input_channels}")
    print(f"- base_num_features: {base_num_features}")
    print(f"- num_classes: {num_classes}")
    print(f"- max_num_features: {max_num_features}")  
    print(f"- num_conv_per_stage: {num_conv_per_stage}")
    print(f"- net_pool_per_axis: {net_pool_per_axis}")
    
    # Create network with correct parameters - explicitly set 3D convolutions and turn off tracking stats
    import torch.nn as nn
    
    # Create network with explicit 3D settings
    network = Generic_UNet(
        input_channels=num_input_channels, 
        base_num_features=base_num_features,
        num_classes=num_classes, 
        num_pool=len(net_pool_per_axis), 
        num_conv_per_stage=num_conv_per_stage,
        pool_op_kernel_sizes=net_pool_per_axis,
        conv_kernel_sizes=conv_kernel_sizes,
        upscale_logits=False, 
        convolutional_pooling=True, 
        convolutional_upsampling=True,
        max_num_features=max_num_features,
        deep_supervision=deep_supervision,
        conv_op=nn.Conv3d,  # Explicitly specify 3D convolution
        norm_op=nn.InstanceNorm3d,  # 3D instance normalization
        norm_op_kwargs={'eps': 1e-5, 'affine': True, 'track_running_stats': False}  # Don't track running stats
    )
    
    # Load weights with better error handling and debugger output
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    # Debug output: print a couple of keys from state_dict to see format
    print("State dict keys (sample):", list(state_dict.keys())[:5])
    
    # Check for possible architecture mismatches
    for key in ['seg_outputs.0.weight', 'conv_blocks_context.0.blocks.0.conv.weight']:
        if key in state_dict:
            print(f"Shape of {key} in checkpoint: {state_dict[key].shape}")
            if hasattr(network, "state_dict") and key in network.state_dict():
                print(f"Shape in our model: {network.state_dict()[key].shape}")
    
    # First try loading with strict=True
    try:
        network.load_state_dict(state_dict)
        print("Successfully loaded state_dict with strict=True!")
    except Exception as e:
        print(f"Warning: Strict loading failed: {e}")
        
        # Get compatibility issues
        missing_keys, unexpected_keys = [], []
        try:
            missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)
            print(f"Loaded with strict=False. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            print("Missing key examples:", missing_keys[:5] if missing_keys else "None")
            print("Unexpected key examples:", unexpected_keys[:5] if unexpected_keys else "None")
        except Exception as e2:
            print(f"Non-strict loading also failed: {e2}")
            
            # Try with DataParallel prefix handling
            try:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k
                    if k.startswith('module.'):
                        name = k[7:]  # remove 'module.' prefix
                    new_state_dict[name] = v
                missing_keys, unexpected_keys = network.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded after removing 'module.' prefix. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
            except Exception as e3:
                print(f"All loading attempts failed: {e3}")
                print("Continuing with uninitialized weights...")
    
    # Set device
    network.eval()
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        network.cuda()
    elif device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        network.to('mps')
    else:
        network.cpu()
    
    return network, plans


def run_sliding_window_inference(network, data, patch_size, step_size=0.5, use_gaussian=True, 
                                 device='cpu', mixed_precision=True):
    """
    Simplified sliding window inference for nnUNet models
    
    Args:
        network: The nnUNet model
        data: Input data tensor
        patch_size: Size of patches to use for sliding window
        step_size: Step size as fraction of patch size
        use_gaussian: Whether to use Gaussian weighting
        device: Device to use for inference
        mixed_precision: Whether to use mixed precision
    
    Returns:
        Softmax predictions
    """
    # Simple sliding window implementation that processes the image in patches
    import torch.nn.functional as F
    
    # Get data dimensions
    b, c, x, y, z = data.shape
    
    # Create output tensor
    result = torch.zeros((b, network.num_classes, x, y, z), dtype=torch.float32)
    if device != 'cpu':
        result = result.to(device)
    
    count_map = torch.zeros((b, network.num_classes, x, y, z), dtype=torch.float32)
    if device != 'cpu':
        count_map = count_map.to(device)
    
    # Calculate steps for each dimension
    steps_x = max(1, int(patch_size[0] * step_size))
    steps_y = max(1, int(patch_size[1] * step_size))
    steps_z = max(1, int(patch_size[2] * step_size))

    # Create Gaussian importance map if needed
    if use_gaussian:
        # Create simple Gaussian importance map
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        tmp[tuple(center_coords)] = 1
        from scipy.ndimage.filters import gaussian_filter
        gaussian_importance_map = gaussian_filter(tmp, sigma=patch_size[0] / 8)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map)
        
        # Convert to torch tensor
        gaussian_importance_map = torch.from_numpy(gaussian_importance_map).float()
        if device != 'cpu':
            gaussian_importance_map = gaussian_importance_map.to(device)
        
        # Add batch and channel dimensions
        gaussian_importance_map = gaussian_importance_map.reshape(1, 1, *patch_size)
        
        # Expand to match number of classes
        gaussian_importance_map = gaussian_importance_map.repeat(1, network.num_classes, 1, 1, 1)
        
    # Process each patch
    for x_start in tqdm.tqdm(range(0, x - patch_size[0] + 1, steps_x)):
        for y_start in tqdm.tqdm(range(0, y - patch_size[1] + 1, steps_y)):
            for z_start in tqdm.tqdm(range(0, z - patch_size[2] + 1, steps_z)):
                # Extract patch
                if x_start == 0 and y_start == 0 and z_start == 0:
                    start = time.time()
                x_end = x_start + patch_size[0]
                y_end = y_start + patch_size[1]
                z_end = z_start + patch_size[2]
                
                # Handle boundary patches
                if x_end > x:
                    x_start = max(0, x - patch_size[0])
                    x_end = x
                if y_end > y:
                    y_start = max(0, y - patch_size[1])
                    y_end = y
                if z_end > z:
                    z_start = max(0, z - patch_size[2])
                    z_end = z
                
                # Get patch data
                patch = data[:, :, x_start:x_end, y_start:y_end, z_start:z_end]
                
                # Run inference
                with torch.no_grad():
                    if mixed_precision and device != 'cpu':
                        # Use autocast for mixed precision
                        with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'mps'):
                            logits = network(patch)
                    else:
                        logits = network(patch)
                
                # Handle tuple output (deep supervision case)
                if isinstance(logits, tuple):
                    print("Deep supervision detected - using the first output")
                    # Use only the first output (highest resolution)
                    logits = logits[0]
                
                # Apply softmax
                softmax = F.softmax(logits, dim=1)
                
                # Apply Gaussian weighting
                if use_gaussian:
                    weight = gaussian_importance_map[:, :, :patch.shape[2], :patch.shape[3], :patch.shape[4]]
                    softmax = softmax * weight
                    weights = weight
                else:
                    weights = torch.ones_like(softmax)
                
                # Add to result
                result[:, :, x_start:x_end, y_start:y_end, z_start:z_end] += softmax
                count_map[:, :, x_start:x_end, y_start:y_end, z_start:z_end] += weights
                if x_start == 0 and y_start == 0 and z_start == 0:
                    end = time.time()
                    print("Time taken for first patch: ", end - start, "seconds")
                    print("Estimated completion time: ", (end - start) * steps_x * steps_y * steps_z / 60, " minutes")
    
    # Normalize by count map
    result = result / (count_map + 1e-8)
    
    return result


def run_single_pass_inference(
    network: torch.nn.Module,
    data: np.ndarray,
    device: str,
    do_mirroring: bool = False,
    mirror_axes: tuple = (0, 1, 2),
    mixed_precision: bool = True
) -> np.ndarray:
    """
    Run fully convolutional inference in a single forward pass.
    
    This is the fastest inference method but requires the entire volume to fit in memory.
    Based on _internal_predict_3D_3Dconv from neural_network.py.
    
    Args:
        network: The nnUNet model
        data: Input data tensor
        device: Device to use for inference
        do_mirroring: Whether to use test-time mirroring
        mirror_axes: Axes to mirror for test-time augmentation
        mixed_precision: Whether to use mixed precision
        
    Returns:
        Segmentation output
    """
    import torch.nn.functional as F
    from batchgenerators.augmentations.utils import pad_nd_image
    
    # Start time measurement
    start_time = time.time()
    
    # Convert data to torch tensor
    if device == 'cuda':
        data_tensor = torch.from_numpy(data).cuda(non_blocking=True)
    elif device == 'mps':
        data_tensor = torch.from_numpy(data).to('mps')
    else:  # cpu
        data_tensor = torch.from_numpy(data)
    
    # Add batch dimension if needed
    if len(data_tensor.shape) == 4:  # c, x, y, z
        data_tensor = data_tensor.unsqueeze(0)  # b, c, x, y, z
    
    # The network needs input with shapes divisible by specific factors
    # Get this from the network or use a default value
    divisibility_factor = getattr(network, 'input_shape_must_be_divisible_by', None)
    if divisibility_factor is None:
        # Default for 3D UNet with 5 pool operations: divisible by 2^5 = 32
        divisibility_factor = 32
        print(f"Using default divisibility factor: {divisibility_factor}")
    elif isinstance(divisibility_factor, (list, tuple, np.ndarray)):
        # If it's a list/tuple/array with different values for different dimensions,
        # use the maximum value for all dimensions for simplicity
        divisibility_factor = max([int(x) for x in divisibility_factor])
        print(f"Using maximum divisibility factor from array: {divisibility_factor}")
    
    # Pad the input to be divisible by the required factor
    current_shape = data_tensor.shape[2:]
    new_shape = []
    for dim in current_shape:
        if dim % divisibility_factor == 0:
            new_shape.append(dim)
        else:
            new_shape.append(((dim // divisibility_factor) + 1) * divisibility_factor)
    
    # Only pad if necessary
    if current_shape != tuple(new_shape):
        print(f"Padding from {current_shape} to {new_shape} to ensure divisibility by {divisibility_factor}")
        # Need to convert back to numpy for pad_nd_image
        data_np = data_tensor.cpu().numpy()
        data_np = data_np[0]  # Remove batch dimension for padding
        
        # Use constant padding by default
        data_padded, slicer = pad_nd_image(data_np, new_shape, "constant", {"constant_values": 0}, True, None)
        
        # Convert back to tensor and add batch dimension
        if device == 'cuda':
            data_tensor = torch.from_numpy(data_padded).unsqueeze(0).cuda(non_blocking=True)
        elif device == 'mps':
            data_tensor = torch.from_numpy(data_padded).unsqueeze(0).to('mps')
        else:  # cpu
            data_tensor = torch.from_numpy(data_padded).unsqueeze(0)
    else:
        # No padding needed, we'll use the full output
        slicer = None
        print("No padding needed, dimensions already divisible")
    
    # Setup mixed precision context
    if mixed_precision:
        if torch.cuda.is_available() and device == 'cuda':
            context = lambda: torch.amp.autocast(device_type='cuda')
        elif torch.backends.mps.is_available() and device == 'mps':
            context = lambda: torch.amp.autocast(device_type='mps')
        else:
            context = lambda: torch.amp.autocast(device_type='cpu')
    else:
        from contextlib import nullcontext
        context = nullcontext
    
    # Run inference in a single forward pass
    with context():
        with torch.no_grad():
            print(f"Running single-pass inference on shape {data_tensor.shape}...")
            
            try:
                # Forward pass
                output = network(data_tensor)
                
                # Handle deep supervision output (tuple)
                if isinstance(output, tuple):
                    output = output[0]  # Use the highest resolution output
                
                # Get initial prediction
                softmax = F.softmax(output, dim=1)
                result = softmax
                
                # Implement test-time mirroring if requested
                if do_mirroring:
                    print("Performing test-time mirroring...")
                    # For storing mirror results
                    mirror_idx = 0
                    num_results = 1
                    
                    # Calculate how many mirror configurations we'll have
                    if do_mirroring:
                        num_results = 2 ** len(mirror_axes)
                    
                    # Process each mirroring configuration
                    for m in range(1, num_results):
                        mirror_idx += 1
                        
                        # Simple mirroring along one axis
                        if m == 1 and (2 in mirror_axes):
                            mirrored = torch.flip(data_tensor, (4,))
                            try:
                                mirror_output = network(mirrored)
                                if isinstance(mirror_output, tuple):
                                    mirror_output = mirror_output[0]
                                mirror_softmax = F.softmax(mirror_output, dim=1)
                                result += torch.flip(mirror_softmax, (4,))
                            except Exception as e:
                                if "is not supported on MPS" in str(e):
                                    print(f"MPS doesn't support this operation: {e}")
                                    print("Skipping this mirror configuration")
                                else:
                                    raise
                        
                        # Mirror along axis 1
                        elif m == 2 and (1 in mirror_axes):
                            mirrored = torch.flip(data_tensor, (3,))
                            try:
                                mirror_output = network(mirrored)
                                if isinstance(mirror_output, tuple):
                                    mirror_output = mirror_output[0]
                                mirror_softmax = F.softmax(mirror_output, dim=1)
                                result += torch.flip(mirror_softmax, (3,))
                            except Exception as e:
                                if "is not supported on MPS" in str(e):
                                    print(f"MPS doesn't support this operation: {e}")
                                    print("Skipping this mirror configuration")
                                else:
                                    raise
                        
                        # Mirror along axis 1 and 2
                        elif m == 3 and (1 in mirror_axes) and (2 in mirror_axes):
                            mirrored = torch.flip(data_tensor, (3, 4))
                            try:
                                mirror_output = network(mirrored)
                                if isinstance(mirror_output, tuple):
                                    mirror_output = mirror_output[0]
                                mirror_softmax = F.softmax(mirror_output, dim=1)
                                result += torch.flip(mirror_softmax, (3, 4))
                            except Exception as e:
                                if "is not supported on MPS" in str(e):
                                    print(f"MPS doesn't support this operation: {e}")
                                    print("Skipping this mirror configuration")
                                else:
                                    raise
                        
                        # Mirror along axis 0
                        elif m == 4 and (0 in mirror_axes):
                            mirrored = torch.flip(data_tensor, (2,))
                            try:
                                mirror_output = network(mirrored)
                                if isinstance(mirror_output, tuple):
                                    mirror_output = mirror_output[0]
                                mirror_softmax = F.softmax(mirror_output, dim=1)
                                result += torch.flip(mirror_softmax, (2,))
                            except Exception as e:
                                if "is not supported on MPS" in str(e):
                                    print(f"MPS doesn't support this operation: {e}")
                                    print("Skipping this mirror configuration")
                                else:
                                    raise
                        
                        # Other mirroring combinations...
                        elif m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                            mirrored = torch.flip(data_tensor, (2, 4))
                            try:
                                mirror_output = network(mirrored)
                                if isinstance(mirror_output, tuple):
                                    mirror_output = mirror_output[0]
                                mirror_softmax = F.softmax(mirror_output, dim=1)
                                result += torch.flip(mirror_softmax, (2, 4))
                            except Exception as e:
                                if "is not supported on MPS" in str(e):
                                    print(f"MPS doesn't support this operation: {e}")
                                    print("Skipping this mirror configuration")
                                else:
                                    raise
                        
                        elif m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                            mirrored = torch.flip(data_tensor, (2, 3))
                            try:
                                mirror_output = network(mirrored)
                                if isinstance(mirror_output, tuple):
                                    mirror_output = mirror_output[0]
                                mirror_softmax = F.softmax(mirror_output, dim=1)
                                result += torch.flip(mirror_softmax, (2, 3))
                            except Exception as e:
                                if "is not supported on MPS" in str(e):
                                    print(f"MPS doesn't support this operation: {e}")
                                    print("Skipping this mirror configuration")
                                else:
                                    raise
                        
                        elif m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                            mirrored = torch.flip(data_tensor, (2, 3, 4))
                            try:
                                mirror_output = network(mirrored)
                                if isinstance(mirror_output, tuple):
                                    mirror_output = mirror_output[0]
                                mirror_softmax = F.softmax(mirror_output, dim=1)
                                result += torch.flip(mirror_softmax, (2, 3, 4))
                            except Exception as e:
                                if "is not supported on MPS" in str(e):
                                    print(f"MPS doesn't support this operation: {e}")
                                    print("Skipping this mirror configuration")
                                else:
                                    raise
                    
                    # Average the results from all mirrors
                    result = result / num_results
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    # Provide helpful error message for memory issues
                    raise RuntimeError(
                        f"OUT OF MEMORY ERROR: Failed to process image of shape {data_tensor.shape} in a single pass. "
                        f"Please use --fast mode instead of --fast-single-pass to use sliding window inference, "
                        f"which uses less memory but is slower. Error: {str(e)}"
                    )
                elif "is not supported on MPS" in str(e):
                    print(f"MPS doesn't support this operation: {e}")
                    print("Falling back to CPU...")
                    
                    # Move everything to CPU and try again
                    network = network.cpu()
                    data_tensor = data_tensor.cpu()
                    
                    # Try again with CPU
                    output = network(data_tensor)
                    
                    # Handle deep supervision output
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    # Apply softmax
                    softmax = F.softmax(output, dim=1)
                    result = softmax
                    
                    # We skip mirroring on CPU as it would be too slow
                    print("Skipping mirroring on CPU for performance reasons")
                else:
                    raise

    # Convert to segmentation by taking argmax
    segmentation = torch.argmax(result, dim=1)
    
    # Remove the batch dimension
    segmentation = segmentation[0]
    
    # If we padded the input, we need to crop the output back to the original size
    if slicer is not None:
        # Create proper slicer for the output, which will have one fewer dimension (no channels)
        output_slicer = tuple([slice(0, segmentation.shape[i]) for i in range(len(segmentation.shape) - (len(slicer) - 1))] + slicer[1:])
        segmentation = segmentation[output_slicer]
    
    # Move to CPU and convert to numpy
    segmentation = segmentation.cpu().numpy()
    
    elapsed_time = time.time() - start_time
    print(f"Single-pass inference completed in {elapsed_time:.2f} seconds")
    
    return segmentation


def run_fast_inference(
    network: torch.nn.Module,
    data: np.ndarray,
    patch_size: tuple,
    device: str,
    do_tta: bool = False,
    mixed_precision: bool = True
) -> np.ndarray:
    """
    Run fast inference without storing intermediate softmax outputs.
    
    This is an optimized version of sliding window inference that uses less memory
    by not storing the full softmax output for all patches.
    
    Args:
        network: The nnUNet model
        data: Input data tensor
        patch_size: Size of patches to use for sliding window
        device: Device to use for inference
        do_tta: Whether to use test-time augmentation
        mixed_precision: Whether to use mixed precision
        
    Returns:
        Segmentation output (not softmax)
    """
    import torch.nn.functional as F
    
    # Convert data to torch tensor
    if device == 'cuda':
        data_tensor = torch.from_numpy(data).cuda(non_blocking=True)
    elif device == 'mps':
        data_tensor = torch.from_numpy(data).to('mps')
    else:  # cpu
        data_tensor = torch.from_numpy(data)
    
    # Add batch dimension if needed
    if len(data_tensor.shape) == 4:  # c, x, y, z
        data_tensor = data_tensor.unsqueeze(0)  # b, c, x, y, z
    
    # Get data dimensions
    b, c, x, y, z = data_tensor.shape
    
    # Create output tensor directly for the segmentation (integer labels)
    # We don't store the full softmax to save memory
    seg_result = torch.zeros((b, x, y, z), dtype=torch.float32)
    if device != 'cpu':
        seg_result = seg_result.to(device)
    
    # Count how many predictions we've made for each voxel
    count_map = torch.zeros((b, x, y, z), dtype=torch.float32)
    if device != 'cpu':
        count_map = count_map.to(device)
    
    # Create simple Gaussian importance map
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    tmp[tuple(center_coords)] = 1
    from scipy.ndimage.filters import gaussian_filter
    gaussian_importance_map = gaussian_filter(tmp, sigma=patch_size[0] / 8)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map)
    
    # Convert to torch tensor
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map).float()
    if device != 'cpu':
        gaussian_importance_map = gaussian_importance_map.to(device)
    
    # Compute steps for each dimension
    steps = []
    for dim, ps in enumerate(patch_size):
        # Maximum coordinate (boundary)
        max_step = data_tensor.shape[dim + 2] - ps
        
        # Calculate number of steps based on step size (0.5 * patch_size)
        num_steps = int(np.ceil(max_step / (ps * 0.5))) + 1
        
        # Compute actual step positions
        if num_steps > 1:
            step_size = max_step / (num_steps - 1)
            step_positions = [int(np.round(i * step_size)) for i in range(num_steps)]
        else:
            step_positions = [0]
            
        steps.append(step_positions)
    
    print(f"Computed steps: x={len(steps[0])}, y={len(steps[1])}, z={len(steps[2])}")
    print(f"Total patches to process: {len(steps[0]) * len(steps[1]) * len(steps[2])}")
    
    # Use context manager for mixed precision if available
    if mixed_precision:
        if torch.cuda.is_available() and device == 'cuda':
            context = lambda: torch.amp.autocast(device_type='cuda')
        elif torch.backends.mps.is_available() and device == 'mps':
            context = lambda: torch.amp.autocast(device_type='mps')
        else:
            context = lambda: torch.amp.autocast(device_type='cpu')
    else:
        from contextlib import nullcontext
        context = nullcontext
    
    # Add support for test-time augmentation if enabled
    if do_tta:
        print("Test-time augmentation enabled in fast mode")
        # Common axes for mirroring in nnUNet
        mirror_axes = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
        num_augmentations = len(mirror_axes) + 1  # +1 for original
    else:
        mirror_axes = []
        num_augmentations = 1

    # Process all augmentations
    for aug_idx in range(num_augmentations):
        # Prepare augmented input
        if aug_idx == 0:
            # Original image
            current_data = data_tensor
            print("Processing original image...")
        else:
            # Apply mirroring
            axis = mirror_axes[aug_idx - 1]
            print(f"Processing mirrored image with axis {axis}...")
            current_data = torch.flip(data_tensor, dims=axis)
        
        # Track patch count for progress reporting
        total_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
        current_patch = 0
            
        # Process patches
        start = time.time()
        with context():
            with torch.no_grad():
                for x_start in steps[0]:
                    for y_start in steps[1]:
                        for z_start in steps[2]:
                            current_patch += 1
                            if current_patch % 10 == 0:
                                print(f"Processing patch {current_patch}/{total_patches} (aug {aug_idx+1}/{num_augmentations})")
                            # Calculate end coordinates with boundary checks
                            x_end = min(x_start + patch_size[0], x)
                            y_end = min(y_start + patch_size[1], y)
                            z_end = min(z_start + patch_size[2], z)
                            
                            # Adjust start positions to maintain patch size
                            if x_end - x_start < patch_size[0]:
                                x_start = max(0, x_end - patch_size[0])
                            if y_end - y_start < patch_size[1]:
                                y_start = max(0, y_end - patch_size[1])
                            if z_end - z_start < patch_size[2]:
                                z_start = max(0, z_end - patch_size[2])
                            
                            # Extract patch
                            patch = current_data[:, :, x_start:x_end, y_start:y_end, z_start:z_end]
                            
                            # Run inference
                            try:
                                output = network(patch)
                                
                                # Handle deep supervision output
                                if isinstance(output, tuple):
                                    output = output[0]  # Use the first output (highest resolution)
                                    
                                # Apply softmax and get predicted class immediately
                                # This is a key memory optimization - we don't store the full softmax
                                softmax = F.softmax(output, dim=1)
                                seg = torch.argmax(softmax, dim=1)  # Convert to segmentation directly
                                
                                # Create one-hot encoding for the segment to add to result
                                # This is needed for proper averaging with TTA
                                seg_one_hot = torch.zeros_like(softmax)
                                for batch_idx in range(seg.shape[0]):
                                    for class_idx in range(network.num_classes):
                                        seg_one_hot[batch_idx, class_idx] = (seg[batch_idx] == class_idx).float()
                                
                                # Apply Gaussian weighting for voting
                                weight_map = gaussian_importance_map[:seg_one_hot.shape[2], :seg_one_hot.shape[3], :seg_one_hot.shape[4]]
                                weight_map = weight_map.view(1, 1, *weight_map.shape)  # Add batch and channel dimension
                                weighted_seg = seg_one_hot * weight_map
                                
                                # Convert back to class indices for the result
                                seg_idx = torch.argmax(weighted_seg, dim=1)
                                
                                # Handle flipped results appropriately
                                if aug_idx > 0:
                                    # Flip back the segmentation
                                    seg_idx = torch.flip(seg_idx, dims=mirror_axes[aug_idx - 1])
                                
                                # Extract segmentation and update count
                                # Apply appropriate weights for averaging across augmentations
                                aug_weight = 1.0 / num_augmentations
                                update_weight = (gaussian_importance_map[:seg.shape[1], :seg.shape[2], :seg.shape[3]]).view(1, *weight_map.shape[2:])
                                
                                # For each position, add the segment index with proper weighting
                                seg_result[:, x_start:x_end, y_start:y_end, z_start:z_end] += seg_idx.float() * update_weight * aug_weight
                                count_map[:, x_start:x_end, y_start:y_end, z_start:z_end] += update_weight * aug_weight
                            
                            except Exception as e:
                                if "is not supported on MPS" in str(e):
                                    print(f"MPS unsupported operation detected: {e}")
                                    print("Falling back to CPU for this patch...")
                                    
                                    # Move to CPU for this patch
                                    cpu_network = network.cpu()
                                    cpu_patch = patch.cpu()
                                    
                                    # Process on CPU
                                    output = cpu_network(cpu_patch)
                                    
                                    # Handle deep supervision output
                                    if isinstance(output, tuple):
                                        output = output[0]
                                        
                                    # Apply softmax and get predicted class immediately
                                    softmax = F.softmax(output, dim=1)
                                    seg = torch.argmax(softmax, dim=1)
                                    
                                    # Create one-hot encoding for the segment
                                    seg_one_hot = torch.zeros_like(softmax)
                                    for batch_idx in range(seg.shape[0]):
                                        for class_idx in range(network.num_classes):
                                            seg_one_hot[batch_idx, class_idx] = (seg[batch_idx] == class_idx).float()
                                    
                                    # Get the CPU version of the Gaussian map
                                    cpu_gaussian = gaussian_importance_map.cpu()
                                    weight_map = cpu_gaussian[:seg_one_hot.shape[2], :seg_one_hot.shape[3], :seg_one_hot.shape[4]]
                                    weight_map = weight_map.view(1, 1, *weight_map.shape)
                                    weighted_seg = seg_one_hot * weight_map
                                    
                                    # Convert back to class indices
                                    seg_idx = torch.argmax(weighted_seg, dim=1)
                                    
                                    # Handle flipped results
                                    if aug_idx > 0:
                                        seg_idx = torch.flip(seg_idx, dims=mirror_axes[aug_idx - 1])
                                    
                                    # Move back to original device
                                    if device != 'cpu':
                                        network.to(device)
                                        seg_idx = seg_idx.to(device)
                                    
                                    # Apply appropriate weights
                                    aug_weight = 1.0 / num_augmentations
                                    update_weight = (cpu_gaussian[:seg.shape[1], :seg.shape[2], :seg.shape[3]]).view(1, *weight_map.shape[2:])
                                    if device != 'cpu':
                                        update_weight = update_weight.to(device)
                                    
                                    # Update results
                                    seg_result[:, x_start:x_end, y_start:y_end, z_start:z_end] += seg_idx.float() * update_weight * aug_weight
                                    count_map[:, x_start:x_end, y_start:y_end, z_start:z_end] += update_weight * aug_weight
                                else:
                                    raise
    end = time.time()
    print("Time taken for inference: ", end - start, "seconds")
    
    # Compute final result by dividing by the count map to get weighted average
    # Avoid division by zero
    count_map = torch.clamp(count_map, min=1e-8)
    final_seg = seg_result / count_map
    final_seg = torch.round(final_seg).long()
    
    # Convert to numpy for saving
    if device != 'cpu':
        final_seg = final_seg.cpu()
    final_seg = final_seg.numpy()
    
    # Remove batch dimension
    if final_seg.shape[0] == 1:
        final_seg = final_seg[0]
    
    return final_seg


def run_inference(
    input_path: str,
    output_path: str,
    model_file: str,
    plans_file: str,
    device: str = 'cuda',
    do_tta: bool = False,
    mixed_precision: bool = True,
    fast_mode: bool = False,
    fast_single_pass: bool = False
) -> None:
    """
    Run nnUNet inference on a single file
    
    Args:
        input_path: Path to input .nii.gz file
        output_path: Path to save output segmentation
        model_file: Path to the model file (.model)
        plans_file: Path to the plans.pkl file
        device: Device to run inference on ('cuda', 'cpu', or 'mps')
        do_tta: Whether to use test-time augmentation
        mixed_precision: Whether to use mixed precision inference
        fast_mode: Whether to use fast inference mode (less memory, potentially faster)
        fast_single_pass: Whether to use single-pass inference (fastest but requires more memory)
    """
    start_time = time.time()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_file}...")
    network, plans = load_network(model_file, plans_file, device)
    
    # Preprocess input image
    print(f"Preprocessing {input_path}...")
    data, properties = load_and_preprocess(input_path, plans)
    
    # Get patch size from plans with fallback options
    if "plans_per_stage" in plans and len(plans["plans_per_stage"]) > 0 and "patch_size" in plans["plans_per_stage"][0]:
        # This is the most reliable source for patch size
        patch_size = plans["plans_per_stage"][0]["patch_size"]
        print(f"Using patch size from plans_per_stage: {patch_size}")
    elif "patch_size" in plans:
        patch_size = plans["patch_size"]
    else:
        # Default patch size for 3D UNet
        # For TotalSegmentator, typical values are around (128, 128, 128)
        print("Warning: Could not find patch_size in plans, using default values (128, 128, 128)")
        patch_size = (128, 128, 128)
    
    # Pick the appropriate inference mode based on flags
    if fast_single_pass:
        print(f"Running FAST SINGLE-PASS inference mode using device: {device}...")
        
        try:
            segmentation = run_single_pass_inference(
                network=network,
                data=data,
                device=device,
                do_mirroring=do_tta,
                mirror_axes=(0, 1, 2),  # Default axes for 3D
                mixed_precision=mixed_precision
            )
            
            # Apply transpose_backward if needed
            if "transpose_backward" in plans:
                transpose_backward = plans["transpose_backward"]
                print(f"Applying transpose_backward: {transpose_backward}")
                try:
                    segmentation = segmentation.transpose([i for i in transpose_backward])
                except Exception as e:
                    print(f"Warning: Error transposing segmentation: {e}")
                    # Skip transpose if it fails
            
            # Save segmentation directly (not using softmax)
            from batchgenerators.augmentations.utils import resize_segmentation
            from nnunet.inference.segmentation_export import save_segmentation_nifti
            
            print(f"Saving segmentation to {output_path}...")
            save_segmentation_nifti(
                segmentation,
                output_path,
                properties,
                order=1,
                force_separate_z=None,
                order_z=0
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nERROR: Single-pass inference failed due to memory constraints: {e}")
                print("Falling back to fast mode (sliding window inference)\n")
                # Set fast_mode to True to fall back to the next fastest option
                fast_mode = True
            else:
                raise
    
    # Use fast inference mode if requested or as fallback from fast_single_pass
    if fast_mode:
        print(f"Running FAST inference mode using device: {device}...")
        
        segmentation = run_fast_inference(
            network=network,
            data=data,
            patch_size=patch_size,
            device=device,
            do_tta=do_tta,
            mixed_precision=mixed_precision
        )
        
        # Apply transpose_backward if needed
        if "transpose_backward" in plans:
            transpose_backward = plans["transpose_backward"]
            print(f"Applying transpose_backward: {transpose_backward}")
            try:
                segmentation = segmentation.transpose([i for i in transpose_backward])
            except Exception as e:
                print(f"Warning: Error transposing segmentation: {e}")
                # Skip transpose if it fails
        
        # Save segmentation directly (not using softmax)
        from batchgenerators.augmentations.utils import resize_segmentation
        from nnunet.inference.segmentation_export import save_segmentation_nifti
        
        print(f"Saving segmentation to {output_path}...")
        save_segmentation_nifti(
            segmentation,
            output_path,
            properties,
            order=1,
            force_separate_z=None,
            order_z=0
        )
        
    else:
        # Use standard sliding window inference (original code)
        # Transfer to device
        if device == 'cuda':
            data_tensor = torch.from_numpy(data).cuda(non_blocking=True)
        elif device == 'mps':
            data_tensor = torch.from_numpy(data).to('mps')
        else:  # cpu
            data_tensor = torch.from_numpy(data)
        
        # Add batch dimension if needed
        if len(data_tensor.shape) == 4:  # c, x, y, z
            data_tensor = data_tensor.unsqueeze(0)  # b, c, x, y, z
        
        print(f"Running inference using device: {device}...")
        
        # Use sliding window inference
        print("Running sliding window inference...")
        try:
            softmax = run_sliding_window_inference(
                network=network,
                data=data_tensor,
                patch_size=patch_size,
                step_size=0.5,
                use_gaussian=True,
                device=device,
                mixed_precision=mixed_precision
            )
        except Exception as e:
            if "is not supported on MPS" in str(e):
                print(f"MPS unsupported operation detected: {e}")
                print("Falling back to CPU for this operation...")
                # Move network to CPU for this operation
                network.cpu()
                cpu_data = data_tensor.cpu()
                softmax = run_sliding_window_inference(
                    network=network,
                    data=cpu_data,
                    patch_size=patch_size,
                    step_size=0.5,
                    use_gaussian=True,
                    device="cpu",
                    mixed_precision=False  # No mixed precision on CPU
                )
                # Move network back to original device if needed
                if device != "cpu":
                    network.to(device)
            else:
                raise
        
        # Test-time augmentation if enabled
        if do_tta:
            print("Performing test-time augmentation...")
            # Common axes for mirroring in nnUNet
            mirror_axes = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
            
            num_mirrors = len(mirror_axes) + 1  # +1 for the original image
            
            for axis in mirror_axes:
                # Flip the input
                flipped = torch.flip(data_tensor, dims=axis)
                
                # Run inference with fallback to CPU if needed
                try:
                    flip_softmax = run_sliding_window_inference(
                        network=network,
                        data=flipped,
                        patch_size=patch_size,
                        step_size=0.5,
                        use_gaussian=True,
                        device=device,
                        mixed_precision=mixed_precision
                    )
                except Exception as e:
                    if "is not supported on MPS" in str(e):
                        print(f"MPS unsupported operation in TTA: {e}")
                        print(f"Falling back to CPU for mirroring axis {axis}...")
                        # Use CPU for this operation
                        if device != "cpu":
                            network.cpu()
                        flipped_cpu = flipped.cpu()
                        flip_softmax = run_sliding_window_inference(
                            network=network,
                            data=flipped_cpu,
                            patch_size=patch_size,
                            step_size=0.5,
                            use_gaussian=True,
                            device="cpu",
                            mixed_precision=False
                        )
                        # Move network back if needed
                        if device != "cpu":
                            network.to(device)
                            flip_softmax = flip_softmax.to(device)
                    else:
                        raise
                
                # Flip back and add to ensemble
                softmax += torch.flip(flip_softmax, dims=axis)
            
            # Average the predictions
            softmax = softmax / num_mirrors
        
        # Convert to numpy for saving
        if device != 'cpu':
            softmax = softmax.cpu()
        softmax = softmax.numpy()
        
        # Get first batch if batched
        if softmax.shape[0] > 1:
            softmax = softmax[0]
        
        # Apply transpose_backward if needed
        if "transpose_backward" in plans:
            transpose_backward = plans["transpose_backward"]
            print(f"Applying transpose_backward: {transpose_backward}")
            try:
                softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])
            except Exception as e:
                print(f"Warning: Error transposing softmax: {e}")
                if isinstance(softmax, np.ndarray):
                    print(f"softmax shape before transpose: {softmax.shape}")
                # Skip transpose if it fails
        
        # Save results
        print(f"Saving segmentation to {output_path}...")
        save_segmentation_nifti_from_softmax(
            softmax, 
            output_path, 
            properties,
            order=1,
            region_class_order=None,
            force_separate_z=None,
            interpolation_order_z=0
        )
    
    elapsed_time = time.time() - start_time
    print(f"Inference complete! Processing time: {elapsed_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Ultra-Simple nnUNet inference CLI")
    
    # Required arguments
    parser.add_argument('--device', type=str, required=True, choices=['cuda', 'cpu', 'mps'],
                        help='Device to use for inference (cuda, cpu, mps)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input .nii.gz file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (.model)')
    parser.add_argument('--plans', type=str, required=True,
                        help='Path to plans.pkl file')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output .nii.gz file (default: input_file_pred.nii.gz)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation (faster)')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision')
    
    # Inference mode selection
    inference_mode = parser.add_mutually_exclusive_group()
    inference_mode.add_argument('--fast', action='store_true',
                        help='Use fast inference mode with sliding window (less memory usage)')
    inference_mode.add_argument('--fast-single-pass', action='store_true',
                        help='Use fully convolutional single-pass inference (fastest, requires more memory)')
    
    args = parser.parse_args()
    
    # Default output name if not specified
    if args.output is None:
        base_name = os.path.basename(args.input)
        name_without_ext = os.path.splitext(base_name)[0]
        if name_without_ext.endswith('.nii'):
            name_without_ext = os.path.splitext(name_without_ext)[0]
        args.output = os.path.join(os.path.dirname(args.input), f"{name_without_ext}_pred.nii.gz")
    
    # Run inference
    run_inference(
        input_path=args.input,
        output_path=args.output,
        model_file=args.model,
        plans_file=args.plans,
        device=args.device,
        do_tta=not args.no_tta,
        mixed_precision=not args.no_mixed_precision,
        fast_mode=args.fast,
        fast_single_pass=args.fast_single_pass
    )


if __name__ == "__main__":
    main()