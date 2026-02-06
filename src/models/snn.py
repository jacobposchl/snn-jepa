'''
Implementation of snn model, using snntorch
'''

import torch
import torch.nn as nn
import snntorch as snn
from typing import Tuple, Optional

class SNNEncoder():
    """ 
    SNN that inputs neural spike data and outputs latent representations
    """

    def __init__(self, input_dim : int, hidden_dims : int,
                 latent_dim : int, timesteps : int,
                 beta : float, threshold : float,
                 homeostatic_target : float):
        
        super().__init__()

    def forward(self, spike_input : torch.Tensor) -> Tuple[torch.Tensor, dict]:
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

        pass

    def compute_homeostatic_penalty(self, metrics : dict) -> torch.Tensor:
        """

        Calculate the penalty by ensuring firing rates are close to bounds

        """

        """
        Something like this:

        penalties = firing rates - homeostatic target
        
        return penalties.mean()
        
        """


        pass