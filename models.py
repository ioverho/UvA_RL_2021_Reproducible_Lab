import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import Adam

import numpy as np

class RELuMLP(nn.Module):
    """
    Simple MLP with ReLU activations.
    """

    def __init__(self, num_inputs, num_actions, hidden):
        super(RELuMLP, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        dims = [self.num_inputs] + (hidden if isinstance(hidden, list) else [hidden]) \
            + [self.num_actions]

        modules = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            modules.append(nn.Linear(d_in, d_out))
            if i != len(dims) - 2:
                modules.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*modules)

    def forward(self, x):

        logits = self.model(x)

        return logits


class ActorDiscrete(nn.Module):
    """
    Actor with discrete actions.
    Should have the following methods:
        - forward. Maps state to action probability distribution
        - sample_action. Samples action based on state
        - compute_trpo_loss. Compute the surrogate TRPO loss and KLD
        - apply_update. Apply update via flattened gradient to all parameters
    """

    def __init__(self, num_inputs, num_actions, hidden):
        super(ActorDiscrete, self).__init__()

        self.model = RELuMLP(num_inputs, num_actions, hidden)

    def forward(self, s, clamp=False):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)

        logits = self.model(s)

        pi_a_s = F.softmax(logits, dim=-1)

        if clamp:
            pi_a_s = torch.distributions.utils.clamp_probs(pi_a_s)

        return pi_a_s

    def sample_action(self, s):

        with torch.no_grad():
            pi_a_s = self.forward(s)

        if hasattr(s, '__len__') and hasattr(s[0], '__len__'):
            num_samples = len(s)
        else:
            num_samples = 1

        action = torch.multinomial(pi_a_s, num_samples)

        action = action.item()

        return action

    def compute_trpo_loss(self, probs_new, probs_old, actions, advantages):

        pi_a_s_old = D.Categorical(probs=probs_old.detach())
        p_a_old = torch.exp(pi_a_s_old.log_prob(actions))

        pi_a_s = D.Categorical(probs=probs_new)
        p_a = torch.exp(pi_a_s.log_prob(actions))

        L = (p_a / p_a_old * advantages).mean()

        KLD = torch.mean(D.kl.kl_divergence(pi_a_s_old, pi_a_s))

        return L, KLD

    def apply_update(self, flattened_grad):
        n = 0
        for p in self.parameters():
            numel = p.numel()
            g = flattened_grad[n:n + numel].view(p.shape)
            p.data += g
            n += numel

    def policy_entropy(self, probs):

        pi_a_s = D.Categorical(probs=probs)

        H_pi_a_s = torch.mean(pi_a_s.entropy())

        return H_pi_a_s


class Critic(nn.Module):
    """Critic using simple MLP as approximator.
    """

    def __init__(self, num_inputs, hidden):
        super(Critic, self).__init__()

        self.model = RELuMLP(num_inputs, 1, hidden)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)

        logits = self.model(s)

        return logits
