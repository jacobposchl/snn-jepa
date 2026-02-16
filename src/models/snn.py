'''
Implementation of snn model, using snntorch
'''

import torch
import torch.nn as nn
import snntorch as snn
from typing import Tuple, Optional

class SNNEncoder(nn.Module):
    """ 
    SNN that inputs neural spike data and outputs latent representations
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list[int], #list of hidden layer sizes
        latent_dim: int,  #size of final output latent vector
        timesteps: int, #for each forward pass
        beta: float = 0.5, #membrane decay rate
        threshold: float = 1.0, #threshold at which neuron fires spike
        homeostatic_target: float = 0.01, #avg firing rate we want
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.homeostatic_target = homeostatic_target

        # build fully connected -> LIF layers (changing back and forth)
        self.fc_layers = nn.ModuleList() #fully connected
        self.lif_layers = nn.ModuleList() #leaky integrate-and-fire

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.fc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.lif_layers.append(
                snn.Leaky(beta=beta, threshold=threshold, learn_beta=True)
                # decay rate can be adjusted during training (True)
            )

        # no spiking here (otherwise), latent space output
        self.latent_out = nn.Linear(hidden_dims[-1], latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim) #normalize for training

    def forward( #called everytime pass data through network
        self, 
        spike_input: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Input the binned spike trains and outputs the latent representation
        """

        """ 
        Should return something like :
        metrics = {
        'firing_rates' : firing_rates
        'spikes' : spikes
        }
        """

        # run SNN model over "timesteps"
        if spike_input.dim() == 3 and spike_input.shape[1] == self.timesteps:
            #rearrange dims from (batch,timesteps,input_dim) to (timesteps,batch,input_dim)
            spike_input = spike_input.permute(1, 0, 2)
        
        t_steps, batch_size, _ = spike_input.shape

        #initial membrane potentials (one per LIF layer) at 0
        mem_states = [lif.init_leaky() for lif in self.lif_layers]

        spike_records = [[] for _ in self.lif_layers]
        hidden_acc = [] #accumulate last layer's activity at each timestep

        for t in range(t_steps):
            x = spike_input[t] # (batch, input_dim)
            for l in range(len(self.fc_layers)):
                x = self.fc_layers[l](x) # (batch, hidden_dim)
                #pass fc output and curr membrane state into LIF neuron
                #neuron spiked = 1, otherwise 0
                spk, updated_mem = self.lif_layers[l](x, mem_states[l])
                mem_states[l] = updated_mem #use updated membrane in next timestep
                spike_records[l].append(spk) #save for firing rate calc
                x = spk #pass spikes as input to the next layer
        
        hidden_acc.append(x) #x is now last-layer spikes

        #after time-loop, create final latent output
        hidden_mean = torch.stack(hidden_acc, dim=0).mean(dim=0)
        latent = self.latent_out(hidden_mean) # (batch, latent_dim)
        latent = self.out_norm(latent)

        firing_rates = [
                torch.stack(spike_records[l], dim=0).mean(dim=0)
                for l in range(len(self.lif_layers))
        ]
        # (n_layers, batch, hidden_dim)

        spikes = [
            torch.stack(spike_records[l], dim=0)
            for l in range(len(self.lif_layers))
        ] #list of (T, batch, hidden_dim)

        metrics = {
            "firing_rates": firing_rates,
            "spikes": spikes,
        }

        # return latent representation
        return latent, metrics

    def compute_homeostatic_penalty(self, metrics : dict) -> torch.Tensor:
        """

        Calculate the penalty by ensuring firing rates are close to bounds

        """

        """
        Something like this:

        penalties = firing rates - homeostatic target
        
        return penalties.mean()
        
        """

        firing_rates = metrics["firing_rates"]  # (n_layers, batch, hidden_dim)
        
        penalties = []
        for f_rate in firing_rates:
            deviation = f_rate - self.homeostatic_target
            penalties.append((deviation ** 2).mean())
        #how far firing rate is from target (positive = too much, negative = too small)
        return torch.stack(penalties).mean()