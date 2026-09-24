"""
Handling different datasets.

Copyright (c) 2026 Madelyn Cruz and Daniel Forger
University of Michigan. All rights reserved.
"""

import torch
from torch.distributions.exponential import Exponential
from torch.utils.data import Dataset
from os.path import exists
from pathlib import Path
import matplotlib.pyplot as plt

from torch import nn

from muutils.json_serialize import serializable_dataclass, SerializableDataclass, serializable_field
from zanj.torchutil import ConfiguredModel, set_config_class

torch.set_default_dtype(torch.float32)

@serializable_dataclass(kw_only=True)
class DatasetConfig(SerializableDataclass):
    """Config for dataset MNIST"""        
    
    sim_t: int = serializable_field(default=14000)
    poisson_max_firings_per: int = serializable_field(default=10)
    poisson_n_timesteps_spike: int = serializable_field(default=100)
    
    train_offset: int = serializable_field(default=0)
    n_samples_test: int = serializable_field(default=900)
    n_samples_val: int = serializable_field(default=100)
    n_samples_train: int = serializable_field(default=200)       
 
# For MNIST Handwritten Digits
def to_spiketrain (output, sample, total_timesteps, max_firings, n_timesteps_spike):
    for pix_id, s in enumerate(sample.flatten()):
        # Sample - 28x28 tensor from 0 to 1
        # Output - Spiketrain representation total_timesteps x (28x28)
        if s < 0.01:
            continue # No spikes, pixel is black.
            
        rate = (max_firings * s) / total_timesteps
        exp = Exponential(rate)
        i = 0
        while i < total_timesteps:
            period = exp.sample() #Sample from the exponential distribution with rate = rate
            i += int(period) # Spike at i + period
            end_pt = min(total_timesteps, i+n_timesteps_spike)
            #print("i, end_pt, period", i, end_pt, period)
            output[i:end_pt, pix_id] = 1.0 
            i = end_pt
