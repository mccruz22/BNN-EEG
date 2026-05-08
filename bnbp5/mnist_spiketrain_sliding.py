import torch
from torch.distributions.exponential import Exponential
from torch.utils.data import Dataset
from os.path import exists
import matplotlib.pyplot as plt

from torch import nn
from typing import Type
from jaxtyping import Float

from muutils.json_serialize import serializable_dataclass, SerializableDataclass, serializable_field
from zanj import ZANJ
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
    #print("sample shape", sample.shape)
    #print("output shape", output.shape)
    for pix_id, s in enumerate(sample):     
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
            

               # xtestspiketrains [sample, i:end_pt, pix_id] = 1.0 

class SpikeTrainMNIST(Dataset):
    """
    Gives the input and labels to neural network (spiketrains and corresponding labels)
    """
    
    # phase should be one of 'test', 'train', 'validation'
    def __init__(self, mnist_dset, phase, CFG2):
        offset = 0
        if phase == 'test':
            n_samples = CFG2.n_samples_test
            offset = CFG2.n_samples_val # Split testing data into (validation U testing) disjoint union.
        elif phase == 'train':
            offset = CFG2.train_offset
            n_samples = CFG2.n_samples_train
        elif phase == 'validation':
            n_samples = CFG2.n_samples_val
        else:
            print(f'ERROR: Invalid phase for MNIST data: {phase}')
            raise ValueError(phase)
        print(f'Loading spiketrains for phase: {phase}, n_samples = {n_samples}, offset = {offset}')
        
        # If in validation, use first n_samples_val
        self.spiketrains = torch.zeros((n_samples, CFG2.sim_t, 28*28))
        self.labels = torch.nn.functional.one_hot(mnist_dset.targets[offset:], num_classes=10) * 1.0
        fname = '../data/spiketrains'
        fname += '_' + phase
        fname += '_' + str(offset)
        fname += '_' + str(n_samples)
        fname += '_' + str(CFG2.sim_t)
        fname += '_' + str(CFG2.poisson_max_firings_per)        
        fname += '_' + str(CFG2.poisson_n_timesteps_spike)
        fname += '.pt'
        
        print(fname)
                
        if exists(fname):
            self.spiketrains = torch.load(fname)    
        else:
            for i in range(n_samples):
                img = mnist_dset[offset + i]
                if (i+1) % 500 == 0:
                    print("%.1f%%" % (100 * i / n_samples))
                # if i < 10:
                #     plt.imshow(img[0][0, :, :])
                #     plt.title(i) 
                #     print(i, self.labels[i])
                #     plt.show()
                to_spiketrain(self.spiketrains[i, :, :], img[0][0, :, :], CFG2.sim_t, CFG2.poisson_max_firings_per, CFG2.poisson_n_timesteps_spike)
            torch.save(self.spiketrains, fname)

    def __len__(self):
        return self.spiketrains.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.spiketrains[idx, :, :], self.labels[idx, :]



class MNISTBrain(Dataset):
    """
    Gives the input and labels to neural network (spiketrains and corresponding labels)
    """
    
    # phase should be one of 'test', 'train', 'validation'
    def __init__(self, inputdata, labels, phase, CFG2, repeat = False):        
        offset = 0
        if phase == 'test':
            n_samples = CFG2.n_samples_test
            offset = CFG2.n_samples_val # Split testing data into (validation U testing) disjoint union.
        elif phase == 'train':
            offset = CFG2.train_offset
            n_samples = CFG2.n_samples_train
        elif phase == 'validation':
            n_samples = CFG2.n_samples_val
        else:
            print(f'ERROR: Invalid phase for MNIST data: {phase}')
            raise ValueError(phase)
        print(f'Loading spiketrains for phase: {phase}, n_samples = {n_samples}, offset = {offset}')
        
        if repeat == False:
            self.spiketrains = inputdata[:n_samples,:,:]
            print("spiketrains", self.spiketrains.shape)

        else:
            # reshaping to adjust for time
            le = 10#100#400#300#200
            self.spiketrains = inputdata[:n_samples, 0:le, :]
            tdt = 10
            for i in range (1, (500-le)//tdt):
                self.spiketrains = torch.cat((self.spiketrains, inputdata[:n_samples, tdt*i:tdt*i+le, :]), dim=-1)
            self.spiketrains =  self.spiketrains.repeat(1,2000//le,1)
           
        
        print("labelsshape", labels.shape)
        self.labels = labels[:n_samples,:] 
        print("labelsshape", self.labels.shape)
        
        print("labels", torch.sum(self.labels, axis=0))
        

    def __len__(self):
        return self.spiketrains.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.spiketrains[idx, :, :], self.labels[idx, :]
    
    
    