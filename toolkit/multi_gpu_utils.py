"""
Multi-GPU utilities for AI Toolkit
Provides device detection, configuration, and validation for multi-GPU training
"""

import os
import torch
import subprocess
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    index: int
    name: str
    memory_total: int  # in MB
    memory_free: int   # in MB
    memory_used: int   # in MB
    utilization: int   # percentage
    temperature: int   # celsius
    power_draw: float  # watts
    power_limit: float # watts


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training"""
    gpu_indices: List[int]
    strategy: str  # 'ddp', 'fsdp', 'auto'
    memory_fraction: float = 0.9
    sync_batchnorm: bool = True
    find_unused_parameters: bool = False


def get_available_gpus() -> List[GPUInfo]:
    """
    Get information about all available GPUs
    Returns a list of GPUInfo objects
    """
    if not torch.cuda.is_available():
        return []
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            # Get basic info from PyTorch
            props = torch.cuda.get_device_properties(i)
            
            # Get detailed info from nvidia-smi if available
            memory_info = get_gpu_memory_info(i)
            utilization_info = get_gpu_utilization_info(i)
            power_info = get_gpu_power_info(i)
            
            gpu_info = GPUInfo(
                index=i,
                name=props.name,
                memory_total=props.total_memory // (1024 * 1024),  # Convert to MB
                memory_free=memory_info.get('free', 0),
                memory_used=memory_info.get('used', 0),
                utilization=utilization_info.get('gpu', 0),
                temperature=utilization_info.get('temperature', 0),
                power_draw=power_info.get('draw', 0.0),
                power_limit=power_info.get('limit', 0.0)
            )
            gpus.append(gpu_info)
        except Exception as e:
            print(f"Warning: Could not get info for GPU {i}: {e}")
            continue
    
    return gpus


def get_gpu_memory_info(gpu_index: int) -> Dict[str, int]:
    """Get memory information for a specific GPU using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.free,memory.used',
            f'--id={gpu_index}',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        free, used = result.stdout.strip().split(', ')
        return {
            'free': int(free),
            'used': int(used)
        }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return {'free': 0, 'used': 0}


def get_gpu_utilization_info(gpu_index: int) -> Dict[str, int]:
    """Get utilization information for a specific GPU using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,temperature.gpu',
            f'--id={gpu_index}',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpu_util, temp = result.stdout.strip().split(', ')
        return {
            'gpu': int(gpu_util),
            'temperature': int(temp)
        }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return {'gpu': 0, 'temperature': 0}


def get_gpu_power_info(gpu_index: int) -> Dict[str, float]:
    """Get power information for a specific GPU using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=power.draw,power.limit',
            f'--id={gpu_index}',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        draw, limit = result.stdout.strip().split(', ')
        return {
            'draw': float(draw) if draw != '[Not Supported]' else 0.0,
            'limit': float(limit) if limit != '[Not Supported]' else 0.0
        }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return {'draw': 0.0, 'limit': 0.0}


def validate_multi_gpu_setup(gpu_indices: List[int], required_memory_per_gpu: int = 0) -> Tuple[bool, List[str]]:
    """
    Validate that the specified GPUs are available and have sufficient memory
    
    Args:
        gpu_indices: List of GPU indices to use
        required_memory_per_gpu: Minimum memory required per GPU in MB
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
        return False, errors
    
    available_gpus = get_available_gpus()
    available_indices = [gpu.index for gpu in available_gpus]
    
    for gpu_idx in gpu_indices:
        if gpu_idx not in available_indices:
            errors.append(f"GPU {gpu_idx} is not available")
            continue
        
        gpu_info = next(gpu for gpu in available_gpus if gpu.index == gpu_idx)
        
        if required_memory_per_gpu > 0 and gpu_info.memory_free < required_memory_per_gpu:
            errors.append(f"GPU {gpu_idx} has insufficient memory: {gpu_info.memory_free}MB free, {required_memory_per_gpu}MB required")
        
        if gpu_info.utilization > 80:
            errors.append(f"GPU {gpu_idx} is heavily utilized ({gpu_info.utilization}%)")
    
    return len(errors) == 0, errors


def get_optimal_gpu_strategy(gpu_indices: List[int], model_size_gb: float = 0) -> str:
    """
    Determine the optimal multi-GPU strategy based on available GPUs and model size
    
    Args:
        gpu_indices: List of GPU indices to use
        model_size_gb: Estimated model size in GB
    
    Returns:
        Recommended strategy: 'ddp', 'fsdp', or 'auto'
    """
    if not gpu_indices:
        return 'ddp'
    
    available_gpus = get_available_gpus()
    gpu_infos = [gpu for gpu in available_gpus if gpu.index in gpu_indices]
    
    if not gpu_infos:
        return 'ddp'
    
    # Calculate average memory per GPU
    avg_memory_gb = sum(gpu.memory_total for gpu in gpu_infos) / len(gpu_infos) / 1024
    
    # If model is large relative to single GPU memory, use FSDP
    if model_size_gb > avg_memory_gb * 0.8:
        return 'fsdp'
    
    # For smaller models or when memory is sufficient, use DDP
    return 'ddp'


def create_multi_gpu_config(
    gpu_indices: Optional[List[int]] = None,
    strategy: str = 'auto',
    memory_fraction: float = 0.9,
    model_size_gb: float = 0
) -> MultiGPUConfig:
    """
    Create a multi-GPU configuration
    
    Args:
        gpu_indices: List of GPU indices to use. If None, uses all available GPUs
        strategy: Multi-GPU strategy ('ddp', 'fsdp', 'auto')
        memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        model_size_gb: Estimated model size in GB for strategy selection
    
    Returns:
        MultiGPUConfig object
    """
    if gpu_indices is None:
        available_gpus = get_available_gpus()
        gpu_indices = [gpu.index for gpu in available_gpus]
    
    if strategy == 'auto':
        strategy = get_optimal_gpu_strategy(gpu_indices, model_size_gb)
    
    return MultiGPUConfig(
        gpu_indices=gpu_indices,
        strategy=strategy,
        memory_fraction=memory_fraction,
        sync_batchnorm=True,
        find_unused_parameters=False
    )


def setup_multi_gpu_environment(config: MultiGPUConfig) -> Dict[str, any]:
    """
    Set up the environment for multi-GPU training
    
    Args:
        config: MultiGPUConfig object
    
    Returns:
        Dictionary with environment variables and settings
    """
    env_vars = {}
    
    # Set CUDA device order
    env_vars['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Set visible devices
    env_vars['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_indices))
    
    # Set PyTorch distributed settings
    if len(config.gpu_indices) > 1:
        env_vars['MASTER_ADDR'] = 'localhost'
        env_vars['MASTER_PORT'] = '12355'
        env_vars['WORLD_SIZE'] = str(len(config.gpu_indices))
        env_vars['RANK'] = '0'  # Will be set per process
    
    return env_vars


def print_gpu_summary(gpu_indices: Optional[List[int]] = None):
    """Print a summary of available GPUs"""
    gpus = get_available_gpus()
    
    if not gpus:
        print("No GPUs available")
        return
    
    if gpu_indices is not None:
        gpus = [gpu for gpu in gpus if gpu.index in gpu_indices]
    
    print(f"\n{'='*80}")
    print("GPU Summary")
    print(f"{'='*80}")
    print(f"{'Index':<6} {'Name':<30} {'Memory':<15} {'Util':<6} {'Temp':<6} {'Power':<10}")
    print(f"{'-'*80}")
    
    for gpu in gpus:
        memory_str = f"{gpu.memory_used}/{gpu.memory_total}MB"
        power_str = f"{gpu.power_draw:.1f}/{gpu.power_limit:.1f}W"
        print(f"{gpu.index:<6} {gpu.name[:29]:<30} {memory_str:<15} {gpu.utilization}%{'':<2} {gpu.temperature}°C{'':<2} {power_str:<10}")
    
    print(f"{'='*80}\n")