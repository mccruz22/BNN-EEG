# Training on BNNs
from bnbp5.mnist_spiketrain_sliding import *
from bnbp5.bnn_intralayer import *

from torchvision import datasets, transforms
import torch
from torch import nn
from sklearn.model_selection import train_test_split

import time
import numpy as np
import pandas as pd
import scipy
import scipy.io
import h5py
import matplotlib.pyplot as plt
import mne

import os
from os import listdir
from os.path import exists
import glob

from zanj import ZANJ

torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_folder = '/nfs/turbo/lsa-forger/mccruz/Anesthesia'

def create_epochs_from_event(start, duration, event_id, raw, epoch_duration=4, tmin = -0.2, sfreq=500, max_num_epochs=10000000000):
    # Create epochs from events
    tmax = epoch_duration + 0.2
    epochs_list = []
    # Create epochs within the event's duration
    current_start = start/sfreq
    num_epochs = 0
    while (current_start*sfreq + epoch_duration <= start + duration) and (num_epochs <= max_num_epochs):
        print("CURRENT START",current_start * sfreq)
        event_array = np.array([[int(current_start * sfreq), 0, event_id]])
        epoch = mne.Epochs(
            raw,
            event_array,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None
        )
        epochs_list.append(epoch)
        current_start += epoch_duration  # Move to the next segment
        num_epochs +=1 
        print(num_epochs)
    return epochs_list

def equal_class_split(X, y, per_class_samples):
    # Ensure equal number of samples per class
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y

    X_balanced, y_balanced = [], []

    for cls in np.unique(y_np):
        idx = np.where(y_np == cls)[0]
        np.random.shuffle(idx)
        selected = idx[:per_class_samples]
        X_balanced.append(X_np[selected])
        y_balanced.append(y_np[selected])

    X_balanced = torch.tensor(np.concatenate(X_balanced), dtype=torch.float32)
    y_balanced = torch.tensor(np.concatenate(y_balanced), dtype=torch.long)
    return X_balanced, y_balanced

class Trainer:
    """
    Training and Validating, and Measuring sliding gradients for a single sample
    """
    
    def __init__(self, CFG1, CFG2, save=True, pretrained='', 
                 subjects=["UM_7"], 
                 ctrs1=[0], ctrs2=[5], num_classes=2):
            
        # Creating model from the config
        self.model = BNN(CFG1).to(device)
        
        self.CFG1 = CFG1
        self.CFG2 = CFG2
        
        self.subjects=subjects
        self.ctrs1 = ctrs1
        self.ctrs2 = ctrs2
        self.num_classes = num_classes
        if pretrained != '':
            self.load_model_from_file(pretrained)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = CFG1.lr)

        print("Model parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
            
        #self.load_anest() # Loading Dataset
        self.load_mnist()
        
    def load_mnist(self):
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
        train_mnist = datasets.MNIST(
            '../data/mnist_torch/',
            train=True, download=True, transform=transform,
        )
        test_mnist = datasets.MNIST(
            '../data/mnist_torch/',
            train=False, download=True, transform=transform,
        )
        self.train_dataset = None # Deallocate        
        self.val_dataset = None # Deallocate
        self.train_dataset = SpikeTrainMNIST(train_mnist, 'train', self.CFG2)
        self.val_dataset = SpikeTrainMNIST(test_mnist, 'validation', self.CFG2) # or test after validation

    def load_anest(self): 
        ## Loading and Preprocessing Dataset (Anesthesia Dataset), Replace with other dataset
        
        # .sfp file
        file_path = base_folder + '/EGI_ChannelLocations/GSN-HydroCel-128.sfp'

        # Load the montage
        montage = mne.channels.read_custom_montage(file_path)

        # Print montage information
        #print(montage)

        df = pd.read_excel(base_folder + '/McDonnell_Events_Info_Summary_TB_10_27_16_DL090817_UManesthesia.xlsx')#, header=True)

        if "UM_4" in self.subjects:
            i=2            
            max_num_epochs = 10000000
        elif "UM_7" in self.subjects:
            i=3            
            max_num_epochs = 10000000
        elif "UM_8" in self.subjects:
            i=4
            max_num_epochs = 10000
        elif "UM_12" in self.subjects:
            i=6
            max_num_epochs = 10000
        elif "UM_13" in self.subjects:
            i=7
            max_num_epochs = 8000            
        elif "UM_9" in self.subjects:
            i=5
            max_num_epochs = 30
        elif "UM_14" in self.subjects:
            i=8
            max_num_epochs = 30
        elif "UM_18" in self.subjects:
            i=9
            max_num_epochs = 30
        elif "UM_21" in self.subjects:
            i=10
            max_num_epochs = 30
        sfreq = 500

        onsets = []
        durations = []
        event_ids = []

        for j in [1, 3, 5]:
            onsets.append(df.loc[i,j])
            durations.append(df.loc[i,j+1]-df.loc[i,j])
            event_ids.append(df.loc[0,j][:-1])


        onsets.extend([df.loc[i, 7], df.loc[i, 11], df.loc[i, 12], df.loc[i, 13],  df.loc[i, 15]])
        durations.extend([df.loc[i, 11] - df.loc[i, 7], df.loc[i, 12]-df.loc[i, 11],df.loc[i, 13]-df.loc[i, 12],df.loc[i, 15]-df.loc[i, 13],df.loc[i, 16]-df.loc[i, 15]])
        event_ids.extend([df.loc[0, 7],df.loc[0, 11], df.loc[0, 12], df.loc[0, 13], df.loc[0, 15]])

        for j in [16, 20, 22, 26, 28, 32, 34, 38, 40, 44, 46, 50, 52]:
            onsets.append(df.loc[i,j])
            durations.append(df.loc[i,j+1]-df.loc[i,j])
            event_ids.append(df.loc[0,j][:-1])

        onsets = np.array(onsets)
        durations = np.array(durations)
        event_ids_text = event_ids
        event_ids = np.arange(len(event_ids)).astype(int)

        onsets_in_samples = (onsets).astype(int)  # Convert from seconds to sample index
        durations_in_samples = (durations ).astype(int)  # Convert from seconds to sample index
        events = np.column_stack((onsets_in_samples, durations_in_samples , event_ids))

        # Define the folder path
        folder_path = base_folder + '/' + self.subjects[0]
        if "UM_7" in self.subjects:
            folder_path = base_folder + '/UM_7'
        elif "UM_8" in self.subjects:
            folder_path = base_folder + '/UM_8'
        elif "UM_12" in self.subjects:
            folder_path = base_folder + '/UM_12'
        elif "UM_13" in self.subjects:
            folder_path = base_folder + '/UM_13'
        elif "UM_4" in self.subjects:
            folder_path = base_folder + '/UM_4'
        
        # Get all .mat files in the folder
        mat_files = glob.glob(os.path.join(folder_path, '*.mat'))

        # Print the list of .mat files
        print(sorted(mat_files))
        eeg_data_list = []

        for file_path in sorted(mat_files):  # file_path = 'UM_12/MDFA10 20140419 0854.mat'
            print(file_path)
            try:
                # Load the .mat file using scipy
                mat_data = scipy.io.loadmat(file_path)

                # Inspect the keys to find the EEG data
                print(mat_data.keys())

                # Assuming 'eeg_data' contains the EEG data in shape (n_channels, n_times)
                eeg_data = mat_data['EEG'][0, 0][15]  # Replace with the actual key name

                # Append the EEG data to the list
                eeg_data_list.append(eeg_data)

            except NotImplementedError:
                with h5py.File(file_path, 'r') as f:
                    eeg_group = f['EEG']
                    print("Keys in 'EEG' group:", list(eeg_group.keys()))

                    # Assuming 'data' contains the EEG data
                    eeg_data = eeg_group['data'][:]  # Replace 'data' with the actual dataset name

                    # Append the transposed EEG data to the list
                    eeg_data_list.append(eeg_data[:,:128].T)

        try:
            # Concatenate all EEG data along the first axis
            eeg_data_all = np.concatenate(eeg_data_list, axis=1)
        except ValueError:
            eeg_data_all = np.array(eeg_data_list)
        
        # Print the final shape of the combined EEG data
        print("Combined EEG data shape:", eeg_data_all.shape)

        info = mne.create_info(ch_names=montage.ch_names, sfreq=500, ch_types='eeg')

        # Apply the montage to the info object
        info.set_montage(montage)

        # Plot the montage
        mne.viz.plot_montage(montage, show=True);

        info = mne.create_info(ch_names=info['ch_names'], sfreq=info['sfreq'], ch_types='eeg')
        raw = mne.io.RawArray(eeg_data_all, info)#[0,0][15]
        #raw = raw.copy().filter(2,60,verbose=False) #tried before and after filter
        raw.set_montage(montage);
       
        del eeg_data_list, eeg_data, info, mat_files
        
        new_events = []

        for i in range (len(events)):
            print(events[i][0], events[i][1], events[i][2])
            
            start = events[i][0]
            duration = events[i][1]
            label = events[i][2]

            if self.num_classes == 2:
                if label <=4:   
                    while duration > 500:                                
                        new_events.append([start, 0, label])
                        start = start + 500
                        duration = duration - 500
                        
            else:
                while duration > 2000:                                
                    new_events.append([start, 0, label])
                    start = start + 2000
                    duration = duration - 2000

        epochs = mne.Epochs(raw, new_events,tmin=-0.200, tmax=4.20, baseline=None, preload=True)        
  
        epochs.pick_types(eeg=True)

        X = epochs.get_data()
        y = torch.tensor(epochs.events[:,-1], dtype=torch.float32)

        del raw, epochs, eeg_data_all #, all_epochs_list, epochs_list
        
        # Assigining classes
        if self.num_classes == 6:
            print("6")
            y[y<=2] = 0
            y[y==3] = 1
            y[y==4] = 2
            y[(y==5) | (y==6)] = 3
            y[y==7] = 4
            y[y>=8] = 5
            
            # Find indices of 0s and 1s
            indices_0 = np.where(y == 0)[0]
            indices_1 = np.where(y == 1)[0]
            indices_2 = np.where(y == 2)[0]
            indices_3 = np.where(y == 3)[0]
            indices_4 = np.where(y == 4)[0]
            indices_5 = np.where(y == 5)[0]

            # Find the minimum count between the two
            min_count = min(len(indices_0), len(indices_1), len(indices_2), len(indices_3), len(indices_4), len(indices_5))

            # Randomly sample min_count from both 0 and 1 indices
            selected_0 = np.random.choice(indices_0, min_count, replace=False)
            selected_1 = np.random.choice(indices_1, min_count, replace=False)
            selected_2 = np.random.choice(indices_2, min_count, replace=False)
            selected_3 = np.random.choice(indices_3, min_count, replace=False)
            selected_4 = np.random.choice(indices_4, min_count, replace=False)
            selected_5 = np.random.choice(indices_5, min_count, replace=False)

            # Combine the selected indices and shuffle them
            selected_indices = np.concatenate([selected_0, selected_1, selected_2, selected_3, selected_4, selected_5])
            np.random.shuffle(selected_indices)

            # Subset y (and X if necessary)
            y = y[selected_indices]
            X = X[selected_indices, :, :]
            
            
        if self.num_classes == 5:
            print("5")
            y[y<=2] = 0
            y[y==3] = 1
            y[y==4] = 2
            y[(y==5) | (y==6)] = 3
            y[y==7] = 4
            y[y>=8] = 5
            
            X = X[y<=4,:,:]
            y = y[y<=4]
            
        
        if self.num_classes == 2:
            print("2")
            y[y<=2] = 0
            y[y==4] = 1
            y[(y==3) | (y>4)] = 2
            
            X = X[y<=1,:,:]
            y = y[y<=1]
            
            # Find indices of 0s and 1s
            indices_0 = np.where(y == 0)[0]
            indices_1 = np.where(y == 1)[0]

            # Find the minimum count between the two
            min_count = min(len(indices_0), len(indices_1))

            # Randomly sample min_count from both 0 and 1 indices
            selected_0 = np.random.choice(indices_0, min_count, replace=False)
            selected_1 = np.random.choice(indices_1, min_count, replace=False)

            # Combine the selected indices and shuffle them
            selected_indices = np.concatenate([selected_0, selected_1])
            np.random.shuffle(selected_indices)

            # Subset y (and X if necessary)
            y = y[selected_indices]
            X = X[selected_indices, :, :]
            
        y = torch.nn.functional.one_hot(torch.Tensor(y).to(torch.int64), num_classes=int(max(y))+1) * 1.0

        X = X.transpose(1,2,0)


        min_val = np.min(X)
        max_val = np.max(X)

        # Normalize the tensor between 0 and 1
        #X = (X - min_val) / (max_val - min_val)
        X = X/max_val
        #X = (X - np.mean(X))/np.std(X)
        X= np.transpose(X, (2, 1, 0))

        x_train, x_test, y_train, y_test = train_test_split(torch.tensor(X),torch.tensor(y),test_size=0.30,random_state=15)

        print("xshape", x_train.shape)
        self.train_dataset = None # Deallocate        
        self.val_dataset = None # Deallocate
        self.train_dataset = MNISTBrain(x_train, y_train, 'train', self.CFG2)
        self.val_dataset = MNISTBrain(x_test, y_test, 'validation', self.CFG2) # or test after validation
        
        
    def load_model_from_file(self, pretrained):
        self.model.load_state_dict(torch.load(pretrained))
        
    def train(self, epoch=0, batches_val=-1, custom_plotter=None):
        loss_fun = nn.MSELoss().to(device)
  
        loss_record = []
        accuracies = []
        start_time = time.time()
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.CFG1.train_batch_sz, shuffle=True)
        
        batch_idx = 0
           
        for batch, expected in train_loader:       
            print(batch_idx)
            self.optimizer.zero_grad()  
        
            V2_out = self.model(batch[:,:,0:self.CFG1.model_dims[0]].to(device))
            out_avg = torch.mean(V2_out, dim=1)
            
            loss = loss_fun(out_avg, expected.to(device))
            loss_record.append(loss.detach())
                          
            loss.backward()
            
            for W in self.model.Ws:
                nn.utils.clip_grad_norm_(W.weight, 1000.0)
                try:
                    if W.weight.grad is not None:  # Ensure grad is not None
                        W.weight.grad[torch.isnan(W.weight.grad)] = 0.0
                except Exception as e:
                    print(f"Error processing gradient for layer {W}: {e}")
            self.optimizer.step()
       
            batch_idx += 1
            
            
            if batches_val > 0 and batch_idx % batches_val == 0:
                print("training error", self.validate(epoch, batch_idx, False))
            
                print(batch_idx, float(loss.detach()), time.time() - start_time)
                accuracy = self.validate(epoch, batch_idx)
                
                accuracies.append(accuracy)
            torch.cuda.empty_cache()
        return accuracies, loss_record
            
        
    def validate(self, epoch=0, batch_idx=-1, use_val_dataset=True):
        dataset = self.val_dataset if use_val_dataset else self.train_dataset
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.CFG1.test_batch_sz, shuffle=False)
        n_hit = 0
        n_total = 0
        start = time.time()
        for batch, expected in val_loader:
            with torch.no_grad():
                out_avg = torch.mean(self.model(batch[:,:,0:self.CFG1.model_dims[0]].to(device)), dim=1)
                guess = torch.argmax(out_avg, dim=1).cpu()
                labels = torch.argmax(expected, dim=1)
                n_hit += torch.sum(guess == labels)
            start = time.time()
            n_total += batch.shape[0]
            
        print("%f" % (n_hit / n_total * 100.0))
        return float(n_hit / n_total * 100.0)
    
    # This function measures the gradients within a sliding window. 
    def measure_sliding_gradients(self, window_size, filename, stride=-1):
        loss_fun = nn.MSELoss().to(device)

        if stride == -1:
            stride = window_size # By default, slide window with no overlap.
        
        # Pick a random training sample to look at. 
        idx = int(torch.randint(0, len(self.train_dataset), ()))
        print(idx)
        sample, target = self.train_dataset[idx]
        
        sample = sample[:,0:self.CFG1.model_dims[0]]
        target = target.to(device)
        sample = sample.unsqueeze(0)
        
        # Feed forward.
        T2_out, interm_out = self.model(sample.to(device), include_intermediates = True)
        T2_out = T2_out.squeeze() # Shape is [SIM_T, OUTPUT_SIZE].
        window_cnt = int((T2_out.shape[0] - window_size) / stride + 1) 
        avgs = torch.zeros((window_cnt, T2_out.shape[1]), requires_grad=True).to(device)
        print("window count", window_cnt, T2_out.shape)
        for i in range(window_cnt):
            avgs[i, :] = torch.mean(T2_out[i * stride: i * stride+window_size, :], 0)
        
        print("Computing sliding window gradients")       
        
        sliding_grad_1 = torch.zeros(window_cnt, self.CFG1.model_dims[0] * self.CFG1.model_dims[1]).to(device)
        sliding_grad_2 = torch.zeros(window_cnt, self.CFG1.model_dims[1] * self.CFG1.model_dims[2]).to(device)
        
        print(sliding_grad_1.shape)
        print(sliding_grad_2.shape)
        W1s = []
        W2s = []
        partialavs = []
        losses = []
        
        print("windowcnt", window_cnt)
        for i in range(window_cnt):
            print(i)
            self.optimizer.zero_grad()
            loss = loss_fun(avgs[i, :], target)
            loss.backward(retain_graph=True)   
            
             # Gradients dL/dw
            sliding_grad_1[i, :] = self.model.Ws[0].weight.grad[:, :].flatten()
            sliding_grad_2[i, :] = self.model.Ws[1].weight.grad[:, :].flatten()
            
            
            losses.append(loss)            
            partialavs.append(avgs.grad)     
            W1s.append(self.model.Ws[0].weight)
            W2s.append(self.model.Ws[1].weight)
            
        fl_prefix = f'{window_size}_{stride}_networkOut'
        z = ZANJ()
        z.save(
          dict(
            losses = losses,
            avgs = avgs,
            partialavs = partialavs,
            W1s =  W1s,
            W2s =  W2s,
            idx = idx,
            sample = sample,
            target = target,
            window_cnt = window_cnt,
            T2_out=T2_out,
            interm_out = interm_out,
            sliding_grad_1 = sliding_grad_1,
            sliding_grad_2 = sliding_grad_2
          ),
         filename,
        )