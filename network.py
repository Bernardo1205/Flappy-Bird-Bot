"""
This file contains a neural network module for us to
define our actor and critic networks in PPO.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FeedForwardNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        """
        Initialize the network and set up the layers.

        Parameters:
            in_dim - input dimensions as an int
            out_dim - output dimensions as an int

        Return:
            None
        """
        super(FeedForwardNN , self).__init__()

        self.layer1 = nn.Linear(in_dim , 64)
        self.layer2 = nn.Linear(64 , 64)
        self.layer3 = nn.Linear(64 ,out_dim)

    def forward(self , obs):
        """
        Runs a forward pass on the neural network.

        Parameters:
            obs - observation to pass as input

        Return:
            output - the output of our forward pass
        """
        # Action selection (rollout time): the PPO agent feeds the current observation into the actor network to get logits (or parameters)
        # for the action distribution and into the critic to get the state value.
        # That call triggers FeedForwardNN.forward (because actor(obs) calls forward internally).
        # The sampled action and its log-prob are then stored for training.
        if isinstance(obs , np.ndarray):
            obs = torch.tensor(obs , dtype= torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output
