## Biological neural network models.
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn

import snntorch as snn

from jaxtyping import Float
from muutils.json_serialize import serializable_dataclass, SerializableDataclass, serializable_field
from zanj.torchutil import ConfiguredModel, set_config_class

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for HH
_HH_PARAMS: dict[str, float] = {
    "gna": 40.0,
    "gk": 35.0,
    "gl": 0.3, #0.3, 0.4

    "Ena": 55.0,
    "Ek": -77.0,
    "El": -65.0,

    "gm": 0.075,
    "ghca": 0.12,
    "vthresh": -56.2, 
    "tau_max": 0.608,
    "Eca": 55.0, #120
    
    "gs": 0.04,
    "Vs": 0.0,
    "Iapp": 0.5,
    "lat_inhibition": False,
    "beta_n_modified": False,

    "Vt": -3.0,
    "Kp": 8.0,
    "a_d": 1.0,
    "a_r": 0.1,
 
}   

# Parameters for Synapse
_SYNAPSE_PARAMS: dict[str, float] = {
    "refractory_period":20,
    "threshold":0,
    "taus":0.1,#0.0000001 worked?
}
        
@serializable_dataclass(kw_only=True)
class BNNConfig(SerializableDataclass):
    """config for biological neural network"""    
    neuron_model: object = serializable_field(
        default_factory=lambda: model_HH_RS,
        serialization_fn=lambda x: x.__name__,
        loading_fn=lambda x: NEURON_MODELS[x["neuron_model"]],
    )

    dt: float = serializable_field(default=0.01)    
    model_dims: list[int] = serializable_field(default_factory=lambda: [28*28, 100, 10])
    lr: float = serializable_field(default=0.001)
    train_batch_sz: int = serializable_field(default=20)
    test_batch_sz: int = serializable_field(default=100)
    network_type: int = serializable_field(default=0)
    z_weight: int = serializable_field(default=1)
    plot_interm: bool = serializable_field(default=False)
    batchnorm: bool = serializable_field(default=False)
    DNN: bool = serializable_field(default=False)
    DNN_ReLU: bool = serializable_field(default=False)
    file_name: str = serializable_field(default="")
    
    neuron_params: dict[str, float] = serializable_field(
        default_factory = lambda: _HH_PARAMS,
    )
        
    synapse_params: dict[str, float] = serializable_field(
        default_factory = lambda: _SYNAPSE_PARAMS,
    )
        
    @property
    def input_dim(self) -> int:
        return self.model_dims[0]
    
    @property
    def output_dim(self) -> int:
        return self.model_dims[-1]

@serializable_dataclass(kw_only=True)
class NNSConfig(SerializableDataclass):
    """config for other non biological neural networks"""  
    
    lr: float = serializable_field(default=0.001)
    train_batch_sz: int = serializable_field(default=20)
    test_batch_sz: int = serializable_field(default=100)
    
    # BiLSTM
    bilstm_model_dims: list[int] = serializable_field(default_factory=lambda: [128, 32, 128, 64, 6]) # [128, 32, 128, 64, 6] - Anesthesia, [454, 32, 128, 64, 2] - DoD
    bilstm_dropout: float = serializable_field(default=0.3)
        
    # SNN
    snn_model_dims: list[int] = serializable_field(default_factory=lambda: [128, 100, 6])
        # [128, 100, 6] - Anesthesia, [454, 100, 2] - DoD
    snn_beta: float = serializable_field(default=0.95) # 0.95 - Anesthesia, 0.99 - DoD
    
class model_HH_Gap(nn.Module):
    """HH with Gap Junctions"""
    def __init__(self, cfg: BNNConfig, dim: int):
        super().__init__()
        self.dim = dim
        self.dt = cfg.dt
        self.P = cfg.neuron_params
        self.plot_interm = cfg.plot_interm
   
    def forward(self, z: Float[torch.Tensor, "B T self.dim"]) -> Float[torch.Tensor, "B T self.dim"]:
        # batch size, num timesteps
        B: int; T: int
        B, T = z.shape[:2]
        dt = self.dt
        
        # hh parameters
        P = self.P
        gna = P['gna']; gk = P['gk']; gl = P['gl'];
        Ena = P['Ena']; Ek = P['Ek']; El = P['El'];
        gs = P['gs']; Vs = P['Vs']; Iapp = P['Iapp'];
        Vt = P['Vt']; Kp = P['Kp'];
        a_d = P['a_d']; a_r = P['a_r'];  

        # init voltage array and m, n, h 
        V: Float[torch.Tensor, "B T self.dim"] = torch.ones((B, T, self.dim)).to(device)*-70.0
        m: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        n: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        h: Float[torch.Tensor, "B T self.dim"] = torch.ones((B, T, self.dim)).to(device)
        
        pow1 = torch.zeros((B, self.dim)).to(device)
        pow2 = torch.zeros((B, self.dim)).to(device)

        # simulation loop
        for k in range(1, T):
            pow1 = gna * (m[:, k-1, :].clone() ** 3) * h[:, k-1, :].clone()
            pow2 = gk * n[:, k-1, :].clone() ** 4
            
            G_scaled = (dt / 2) * (pow1 + pow2 + gl)
            E = pow1 * Ena + pow2 * Ek + gl * El
            
            V[:, k, :] = (V[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp + z[:, k-1, :])) / (1 + G_scaled)
            
            aN = 0.02 * (V[:, k, :] - 25) / (1 - torch.exp((-V[:, k, :] + 25) / 9.0))
            aM = 0.182 * (V[:, k, :] + 35) / (1 - torch.exp((-V[:, k, :] - 35) / 9.0))
            aH = 0.25 * torch.exp((-V[:, k, :] - 90) / 12.0)
            
            bN = -0.002 * (V[:, k, :] - 25) / (1 - torch.exp((V[:, k, :] - 25) / 9.0))
            if P['beta_n_modified']:
                bN = 0.125 * torch.exp((-V[:, k, :] + 70) / 19.7)
            bM = -0.124 * (V[:, k, :] + 35) / (1 - torch.exp((V[:, k, :] + 35) / 9.0))
            bH = 0.25 * torch.exp((V[:, k, :] + 34) / 12.0)
            
            # zero denominators
            if torch.any(V[:, k, :] == 25) or torch.any(V[:, k, :] == -35):
                aN[torch.where(V[:, k, :] == 25)] = 0.18
                bN[torch.where(V[:, k, :] == 25)] = 0.08
                aM[torch.where(V[:, k, :] == -35)] = 1.638
                bM[torch.where(V[:, k, :] == -35)] = 1.16
                
            m[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
            n[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
            h[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)    
            
            # lateral inhibition
            if P['lat_inhibition']:
                with torch.no_grad():
                    V_sub = V[:, k, :] + 70
                    below = V_sub.le(60)
                    V[:, k, :] = torch.where(below, V_sub * (below.sum(axis=1).reshape(-1,1) / self.L) - 70,  V_sub - 70)

        print("-- HH Gap Working --")
        if self.plot_interm == True:
            plt.figure(figsize=(15,5))
            plt.subplot(1,2,1)
            plt.plot(z[0, :, :].detach().cpu().numpy(), linewidth=1.0)
            plt.xticks(range(0,T+1,500), [f'{i*dt}' for i in range(0,T+1,500)])
            plt.xlabel('Time (ms)', fontsize=14)
            plt.title('A. Weighted Input', fontsize=18)
            plt.subplot(1,2,2)
            plt.plot(V[0, :, :].detach().cpu().numpy(), linewidth=1.0)
            plt.xticks(range(0,T+1,500), [f'{i*dt}' for i in range(0,T+1,500)])
            plt.xlabel('Time (ms)', fontsize=14)
            plt.title('B. Voltage Response', fontsize=18)
            plt.show()
        
        return torch.sigmoid((V - Vt) / Kp)
    
class model_HH_RS(nn.Module):
    """Regular Spiking Neurons""" 
    def __init__(self, cfg: BNNConfig, dim: int):
        super().__init__()
        self.dim = dim
        self.dt = cfg.dt
        self.P = cfg.neuron_params
        self.z_weight = cfg.z_weight
        self.network_type = cfg.network_type
        self.plot_interm = cfg.plot_interm
        self.file_name = cfg.file_name        
        
        # intralayer connections with random initial synaptic weights
        if self.network_type == 1:       
            torch.manual_seed(15)
            self.syn_weights = nn.Parameter(torch.randn(dim, dim))
            self.P_syn = cfg.synapse_params
    
    def forward(self, z: Float[torch.Tensor, "B T self.dim"]) -> Float[torch.Tensor, "B T self.dim"]:
        # batch size, num timesteps
        B: int; T: int
        B, T = z.shape[:2]
        dt = self.dt

        # hh parameters
        P = self.P        
        gna = P['gna']; gk = P['gk']; gl = P['gl']; gm = P['gm'];
        Ena = P['Ena']; Ek = P['Ek']; El = P['El'];
        Iapp = P['Iapp']; Vt = P['Vt']; Kp = P['Kp'];
        vthresh = P['vthresh']; tau_max = P['tau_max'];
        
        # init voltage array and m, n, h, p
        V: Float[torch.Tensor, "B T self.dim"] = torch.ones((B, T, self.dim)).to(device)*-70.0
        m: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        n: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        h: Float[torch.Tensor, "B T self.dim"] = torch.ones((B, T, self.dim)).to(device)
        p: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        
        pow1 = torch.zeros((B, self.dim)).to(device)
        pow2 = torch.zeros((B, self.dim)).to(device)
        powm = torch.zeros((B, self.dim)).to(device)
        
        # inputs weighted (default unweighted)
        z_weight = self.z_weight
        
        if self.network_type >= 1:
            print("-- Network has intralayer connections. --")
            # synaptic parameters
            P_syn = self.P_syn
            
            refractory_period = P_syn['refractory_period']
            threshold = P_syn['threshold']
            taus = P_syn['taus']
            
            spike_times = torch.zeros((z.shape[0], z.shape[2])).to(device)        

        # simulation loop
        for k in range(1, T):
            pow1 = gna * (m[:, k-1, :].clone() ** 3) * h[:, k-1, :].clone()
            pow2 = gk * n[:, k-1, :].clone() ** 4
            powm = gm *p[:, k-1, :].clone() 
            
            G_scaled = (dt / 2) * (pow1 + pow2 + gl + powm)
            E = pow1 * Ena + pow2 * Ek + gl * El + powm * Ek
            
            if self.network_type == 1:
                # vectorized version of code above
                threshold_mask = (V[:, k - 1, :] >= threshold)
                refractory_mask = ((k - 1 - spike_times) > refractory_period)
                valid_spikes_mask = (threshold_mask & refractory_mask)
                spike_times[valid_spikes_mask] = k - 1
                spikes_windowed = torch.exp(-(k-spike_times)/taus) * (spike_times > 0)
                
                synaptic_activity = torch.matmul(spikes_windowed.to(device), self.syn_weights.to(device)) 
                
                V[:, k, :] = (V[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp + z_weight*z[:, k-1, :] + synaptic_activity)) / (1 + G_scaled)  
            
            # original BNN, feedforward network
            else:
                V[:, k, :] = (V[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp + z_weight*z[:, k-1, :])) / (1 + G_scaled)
            
            aH = 0.25 * torch.exp((-V[:, k, :] - 90) / 12.0)
            bH = 0.25 * torch.exp((V[:, k, :] + 34) / 12.0)
            
            # zero denominators
            if torch.any(V[:, k, :] == 25) or torch.any(V[:, k, :] == -35):
                aN[torch.where(V[:, k, :] == 25)] = 0.18
                bN[torch.where(V[:, k, :] == 25)] = 0.08
                aM[torch.where(V[:, k, :] == -35)] = 1.638
                bM[torch.where(V[:, k, :] == -35)] = 1.16
                
            else:
                aN = 0.02 * (V[:, k, :] - 25) / (1 - torch.exp((-V[:, k, :] + 25) / 9.0))
                aM = 0.182 * (V[:, k, :] + 35) / (1 - torch.exp((-V[:, k, :] - 35) / 9.0))
                bN = -0.002 * (V[:, k, :] - 25) / (1 - torch.exp((V[:, k, :] - 25) / 9.0))
                bM = -0.124 * (V[:, k, :] + 35) / (1 - torch.exp((V[:, k, :] + 35) / 9.0))
                
            pinf = 1 / (1 + torch.exp(-(V[:, k, :] + 35) / 10.0))
            taup = tau_max / (3.3 *  torch.exp((V[:, k, :] + 35) / 20.0) +  torch.exp(-(V[:, k, :] + 35) / 20.0))
            
            taup[torch.where(taup == 0)] = tau_max / (3.3 *   +  1 / 20.0)   
            taup[torch.where(taup == -0.5)] = tau_max / (3.3 *   +  1 / 20.0) 
            taup[torch.where(V[:, k, :] == -35)] = tau_max / (3.3 *   +  1 / 20.0)   
            
            m[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
            n[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
            h[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)
            p[:, k, :] = (pinf/taup * dt + (1 - dt / (2 * taup)) * p[:, k-1, :].clone()) /  (dt / (2 * taup) + 1) 
            
            # lateral inhibition
            if P['lat_inhibition']:
                with torch.no_grad():
                    V_sub = V[:, k, :] + 70
                    below = V_sub.le(60)
                    V[:, k, :] = torch.where(below, V_sub * (below.sum(axis=1).reshape(-1,1) / self.L) - 70,  V_sub - 70)
        
        print("-- HH RS Working --")
        if self.plot_interm == True:            
            plt.figure(figsize=(15,5))
            plt.subplot(1,2,1)
            plt.plot(z[0, :, :1].detach().cpu().numpy(), linewidth=1.0)
            plt.xticks(range(0,T+1,500), [f'{i*dt}' for i in range(0,T+1,500)])
            plt.xlabel('Time (ms)', fontsize=14)
            plt.title('A. Weighted Input', fontsize=18)
            plt.subplot(1,2,2)
            plt.plot(V[0, :, :].detach().cpu().numpy(), linewidth=1.0)
            plt.xticks(range(0,T+1,500), [f'{i*dt}' for i in range(0,T+1,500)])
            plt.xlabel('Time (ms)', fontsize=14)
            plt.title('B. Voltage Response', fontsize=18)
            plt.show()
            
            # saving state variables            
            print("SAVING HH")            
            np.savez(f"hh_vars_{self.file_name}_{self.dim}.npz", V=V.cpu().detach().numpy(), m=m.cpu().detach().numpy(), h=h.cpu().detach().numpy(), n=n.cpu().detach().numpy(),p=p.cpu().detach().numpy(), z=z.cpu().detach().numpy())
                
        return torch.sigmoid((V - Vt) / Kp)

class model_HH_IBN(nn.Module):
    """Intrinsically Bursting Neurons"""
    def __init__(self, cfg: BNNConfig, dim: int):
        super().__init__()
        self.dim = dim
        self.dt = cfg.dt
        self.P = cfg.neuron_params
        self.plot_interm = cfg.plot_interm
   
    def forward(self, z: Float[torch.Tensor, "B T self.dim"]) -> Float[torch.Tensor, "B T self.dim"]:
        # batch size, num timesteps
        B: int; T: int
        B, T = z.shape[:2]
        dt = self.dt
        
        # hh parameters
        P = self.P
        gna = P['gna']; gk = P['gk']; gl = P['gl']; gm = P['gm']; ghca = P['ghca']
        Ena = P['Ena']; Ek = P['Ek']; El = P['El']; Eca = P['Eca'];
        Iapp = P['Iapp']; Vt = P['Vt']; Kp = P['Kp'];
        vthresh = P['vthresh']; tau_max = P['tau_max'];
        
        # init voltage array and m, n, h, p, q, r
        V: Float[torch.Tensor, "B T self.dim"] = torch.ones((B, T, self.dim)).to(device)*-70.0
        m: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        n: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        h: Float[torch.Tensor, "B T self.dim"] = torch.ones((B, T, self.dim)).to(device)
        p: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        q: Float[torch.Tensor, "B T self.dim"] = torch.zeros((B, T, self.dim)).to(device)
        r: Float[torch.Tensor, "B T self.dim"] = torch.ones((B, T, self.dim)).to(device)
        
        pow1 = torch.zeros((B, self.dim)).to(device)
        pow2 = torch.zeros((B, self.dim)).to(device)
        powm = torch.zeros((B, self.dim)).to(device)
        powhca = torch.zeros((B, self.dim)).to(device)

        # simulation loop
        for k in range(1, T):
            pow1 = gna * (m[:, k-1, :].clone() ** 3) * h[:, k-1, :].clone()
            pow2 = gk * n[:, k-1, :].clone() ** 4
            powm = gm *p[:, k-1, :].clone() 
            powhca = ghca * (q[:, k-1, :].clone() ** 2) * r[:, k-1, :].clone()
            
            G_scaled = (dt / 2) * (pow1 + pow2 + gl + powm + powhca)
            E = pow1 * Ena + pow2 * Ek + gl * El + powm * Ek + powhca * Eca 
            
            V[:, k, :] = (V[:, k-1, :].clone() * (1 - G_scaled) + dt * (E + Iapp + z[:, k-1, :])) / (1 + G_scaled)
            
            aN = 0.02 * (V[:, k, :] - 25) / (1 - torch.exp((-V[:, k, :] + 25) / 9.0))
            aM = 0.182 * (V[:, k, :] + 35) / (1 - torch.exp((-V[:, k, :] - 35) / 9.0))
            aH = 0.25 * torch.exp((-V[:, k, :] - 90) / 12.0)
            
            bN = -0.002 * (V[:, k, :] - 25) / (1 - torch.exp((V[:, k, :] - 25) / 9.0))
            bM = -0.124 * (V[:, k, :] + 35) / (1 - torch.exp((V[:, k, :] + 35) / 9.0))
            bH = 0.25 * torch.exp((V[:, k, :] + 34) / 12.0)
            
            # aM = 0.32 * (self.V[:, k, :] - vthresh - 13) / (1-  torch.exp(-(self.V[:, k, :] - vthresh - 13)/4))
            # aH = 0.128 * torch.exp(-(self.V[:, k, :] - vthresh - 17) / 18.0)
            # aN = 0.032 * (self.V[:, k, :] - vthresh - 15)/ (1-  torch.exp(-(self.V[:, k, :] - vthresh - 15)/5))
            
            # bM = - 0.28 * (self.V[:, k, :] - vthresh -40) / (1 - torch.exp((self.V[:, k, :]- vthresh -40) / 5.0))
            # bH = 4/ (1 + torch.exp(-(self.V[:, k, :] -vthresh - 40) / 5.0))
            # bN = 0.5 * torch.exp((-self.V[:, k, :] + vthresh + 10) / 40.0)
            
            pinf = 1 / (1 + torch.exp(-(V[:, k, :] + 35) / 10.0))
            taup = tau_max / (3.3 *  torch.exp((V[:, k, :] + 35) / 20.0) +  torch.exp(-(V[:, k, :] + 35) / 20.0))
            
            aQ = 0.055 * (-V[:, k, :] - 27) / (torch.exp((-V[:, k, :] - 27) / 3.8) - 1)
            aR = 0.000457 * torch.exp((-V[:, k, :] - 13) / 50.0)
            
            bQ = 0.94 * torch.exp((-V[:, k, :] - 75) / 17.0)        
            bR = 0.0065 / (torch.exp((-V[:, k, :] - 15) / 28.0) + 1)
            
            # zero denominators
            if torch.any(V[:, k, :] == 25) or torch.any(V[:, k, :] == -35):
                aN[torch.where(V[:, k, :] == 25)] = 0.18
                bN[torch.where(V[:, k, :] == 25)] = 0.08
                aM[torch.where(V[:, k, :] == -35)] = 1.638
                bM[torch.where(V[:, k, :] == -35)] = 1.16
            
            if torch.any(V[:, k, :] == -27):
                aQ[torch.where(V[:, k, :] == 25)] = 0.055*3.8
                
            m[:, k, :] = (aM * dt + (1 - dt / 2 * (aM + bM)) * m[:, k-1, :].clone()) / (dt / 2 * (aM + bM) + 1)
            n[:, k, :] = (aN * dt + (1 - dt / 2 * (aN + bN)) * n[:, k-1, :].clone()) / (dt / 2 * (aN + bN) + 1)
            h[:, k, :] = (aH * dt + (1 - dt / 2 * (aH + bH)) * h[:, k-1, :].clone()) / (dt / 2 * (aH + bH) + 1)
            p[:, k, :] = (pinf/taup * dt + (1 - dt / (2 * taup)) * p[:, k-1, :].clone()) /  (dt / (2 * taup) + 1)
            q[:, k, :] = (aQ * dt + (1 - dt / 2 * (aQ + bQ)) * q[:, k-1, :].clone()) / (dt / 2 * (aQ + bQ) + 1)
            r[:, k, :] = (aR * dt + (1 - dt / 2 * (aR + bR)) * r[:, k-1, :].clone()) / (dt / 2 * (aR + bR) + 1)
            
            # Lateral inhibition
            if P['lat_inhibition']:
                with torch.no_grad():
                    V_sub = V[:, k, :] + 70
                    below = V_sub.le(60)
                    V[:, k, :] = torch.where(below, V_sub * (below.sum(axis=1).reshape(-1,1) / self.L) - 70,  V_sub - 70)
        
        print("-- HH IBN Working --")
        if self.plot_interm == True:
            plt.figure(figsize=(15,5))
            plt.subplot(1,2,1)
            plt.plot(z[0, :, :].detach().cpu().numpy(), linewidth=1.0)
            plt.xticks(range(0,T+1,500), [f'{i*dt}' for i in range(0,T+1,500)])
            plt.xlabel('Time (ms)', fontsize=14)
            plt.title('A. Weighted Input', fontsize=18)
            plt.subplot(1,2,2)
            plt.plot(V[0, :, :].detach().cpu().numpy(), linewidth=1.0)
            plt.xticks(range(0,T+1,500), [f'{i*dt}' for i in range(0,T+1,500)])
            plt.xlabel('Time (ms)', fontsize=14)
            plt.title('B. Voltage Response', fontsize=18)
            plt.show()
        
        return torch.sigmoid((V - Vt) / Kp)
   
NEURON_MODELS: dict[str, object] = {
    "model_HH_Gap": model_HH_Gap,
    "model_HH_RS": model_HH_RS,
    "model_HH_IBN": model_HH_IBN
}

@set_config_class(BNNConfig)
class BNN(ConfiguredModel[BNNConfig]):
    """
    model class for biological neural network, subclass of ConfiguredModel
    """    
    
    @property
    def cfg(self) -> BNNConfig:
        return self.zanj_model_config
    
    def __init__(self, config: BNNConfig):
        # storing the model in the zanj_model_config
        super().__init__(config)
      
        self.Ws: torch.nn.ModuleList[nn.Linear] = nn.ModuleList()
        self.layers: torch.nn.ModuleList[nn.Module] = nn.ModuleList()
        for d1, d2 in zip(self.cfg.model_dims[:-1], self.cfg.model_dims[1:]):
            print(d1, d2)
            self.Ws.append(nn.Linear(d1, d2, bias=False))
            if self.cfg.DNN:
                self.layers.append(nn.Sigmoid())
            elif self.cfg.DNN_ReLU:
                self.layers.append(nn.ReLU()) 
            else:
                self.layers.append(self.cfg.neuron_model(self.cfg, d2))

        # need to add weights as parameters of model.
        for i, W in enumerate(self.Ws):
            self.__setattr__(f'W{i+1}', W)
                
    def forward(
            self, 
            batch: torch.Tensor,
            include_intermediates: bool = False,
        ) -> torch.Tensor:

        intermediates: list[tuple[torch.Tensor, torch.Tensor]] = []
        T = batch.float()
                
        for W, layer in zip(self.Ws, self.layers):
            # normal BNN
            print("T shape",T.shape)
            z = W(T)
            print("z shape",z.shape)
            T = layer(z)

            # normalized per layer
            if self.cfg.batchnorm:
                print("normalizing", T.shape)
                mnormed = nn.BatchNorm1d(T.shape[-1]).to(device)
                T=mnormed(T.transpose(2,1).to(device)).transpose(2,1).to(device)
                print("normalized", T.shape)

            # include z in outputs
            if include_intermediates:
                intermediates.append((z, T))

        if include_intermediates:
            return T, intermediates
        else:
            return T

@set_config_class(NNSConfig)
class LSTMModel(ConfiguredModel[NNSConfig]):
    """
    model class for BiLSTM, subclass of ConfiguredModel
    """    
    
    @property
    def cfg(self) -> NNSConfig:
        return self.zanj_model_config
    
    def __init__(self, config: NNSConfig):
        # storing the model in the zanj_model_config
        super().__init__(config)        
        
        # define layers
        self.dense = nn.Linear(self.cfg.bilstm_model_dims[0], self.cfg.bilstm_model_dims[1])  
        self.lstm = nn.LSTM(self.cfg.bilstm_model_dims[1], self.cfg.bilstm_model_dims[2], bidirectional=True, batch_first=True)  # biLSTM
        self.dropout = nn.Dropout(self.cfg.bilstm_dropout)  
        self.batch_norm1 = nn.BatchNorm1d(self.cfg.bilstm_model_dims[2]*2)  # 128*2 for bidirectional
        self.dense1 = nn.Linear(self.cfg.bilstm_model_dims[2]*2, self.cfg.bilstm_model_dims[3])  
        self.dropout2 = nn.Dropout(self.cfg.bilstm_dropout)  
        self.batch_norm2 = nn.BatchNorm1d(self.cfg.bilstm_model_dims[3])  
        self.output_layer = nn.Linear(self.cfg.bilstm_model_dims[3], self.cfg.bilstm_model_dims[4])  # final output layer (6 classes)
          
    def forward(self, x):
        # forward pass through layers
       
        x = F.relu(self.dense(x.float()))  # linear layer with ReLU
        print("shapesx", x.shape)
        x, (hn, cn) = self.lstm(x)  # LSTM layer
        print("shapesx", x.shape)
        x = self.dropout(x)  # dropout after LSTM
        print("shapesx", x.shape)
        x = self.batch_norm1(x[:, -1, :])  # BatchNorm on the last time step of LSTM output
        print("shapesx", x.shape)
        x = F.relu(self.dense1(x))  # second linear layer with ReLU
        print("shapesx", x.shape)
        x = self.dropout2(x)  # second dropout
        print("shapesx", x.shape)
        x = self.batch_norm2(x)  # second BatchNorm
        print("batchnorm", x.shape)
        #x = F.softmax(self.output_layer(x), dim=1)  # softmax output layer
        x = F.sigmoid(self.output_layer(x))  # sigmoid output layer
        
        print("final", x.shape)
        return x
    
@set_config_class(NNSConfig)
class SNNNet(ConfiguredModel[NNSConfig]):
    """
    model class for spiking neural network with LIF Neurons, subclass of ConfiguredModel
    """    
    
    @property
    def cfg(self) -> NNSConfig:
        return self.zanj_model_config
    
    def __init__(self, config: NNSConfig):
        # storing the model in the zanj_model_config
        super().__init__(config)

        num_inputs = self.cfg.snn_model_dims[0] #128
        num_hidden = self.cfg.snn_model_dims[1] #100
        num_outputs = self.cfg.snn_model_dims[2] #6
        
        beta = self.cfg.snn_beta #0.95

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # record the final layer
        spk2_rec = []
        mem2_rec = []
        
        x=x.float()

        print("XSHAPE", x.shape)
        for step in range(x.shape[1]-1):
            cur1 = self.fc1(x[:,step,:])
            spk1, mem1 = self.lif1(cur1, x[:,step+1,:])
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)       