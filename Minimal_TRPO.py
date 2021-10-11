# 'Inspired' by:
# https://gist.github.com/elumixor/c16b7bdc38e90aa30c2825d53790d217

from collections import namedtuple

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import Adam

import gym

# ==============================================================================
# Models
# ==============================================================================

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
    """Actor with discrete actions.
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

# ==============================================================================
# Train Functions
# ==============================================================================

def estimate_advantages(batch, critic):
    def _estimate_advantage(critic, states, last_state, rewards):
        """Computes the advantages under the policy actions.

        Args:
            states ([type]): [description]
            last_state ([type]): [description]
            rewards ([type]): [description]

        Returns:
            [type]: [description]
        """
        values = critic(states)

        last_value = critic(last_state.unsqueeze(0))
        next_values = torch.zeros_like(rewards)

        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value

        advantages = next_values - values

        return advantages

    advantages = [_estimate_advantage(critic, states, next_states[-1], rewards)
                    for states, _, rewards, next_states in batch]
    advantages = torch.cat(advantages, dim=0).flatten()

    # Normalize advantages to reduce skewness and improve convergence
    advantages = (advantages - torch.mean(advantages)) / \
        torch.std(advantages)

    return advantages


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(
        y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x

# ==============================================================================
# Hyper-parameters
# ==============================================================================

batch_size = 16
max_d_kl = 0.01
actor_hidden = [32]
critic_lr = 5e-3
critic_hidden = [32]
epochs = 50
max_ls_steps = 10
ls_step_coef = 0.9

# ==============================================================================
# Main loop
# ==============================================================================
env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

actor = ActorDiscrete(num_inputs=state_size, num_actions=num_actions, hidden=actor_hidden)

critic = Critic(num_inputs=state_size, hidden=critic_hidden)
critic_optimizer = Adam(critic.parameters(), lr=critic_lr)

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', ])

def train(epochs=100):
    total_rewards_mean = []
    total_rewards_se = []

    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []

        # ======================================================================
        # Generate batched episode
        # ======================================================================
        for _ in range(batch_size):
            state = env.reset()
            done = False

            samples = []

            while not done:
                with torch.no_grad():
                    action = actor.sample_action(state)

                next_state, reward, done, _ = env.step(action)

                samples.append((state, action, reward, next_state))

                state = next_state

            # Transpose samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))

            rollout_total_rewards.append(rewards.sum().item())

        # ======================================================================
        # Use episode to gather states, actions and advantages based on both
        # ======================================================================
        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

        advantages = estimate_advantages(rollouts, critic)

        # ======================================================================
        # Update the critic
        # ======================================================================
        critic_loss = 0.5 * torch.mean(advantages ** 2)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ======================================================================
        # TRPO Actor Loss
        # ======================================================================
        probs = actor.forward(states, clamp=True)

        L, KLD = actor.compute_trpo_loss(probs, probs, actions, advantages)

        # ======================================================================
        # Gather gradients & KL + Fisher matrix for actor update
        # ======================================================================
        parameters = list(actor.parameters())

        g = flat_grad(L, parameters, retain_graph=True)
        d_kl = flat_grad(KLD, parameters, create_graph=True)

        def HVP(v):
            return flat_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        # ======================================================================
        # Line search over possible step sizes
        # ======================================================================
        for ii in range(max_ls_steps):

            proposal_update = (ls_step_coef ** ii) * max_step

            actor.apply_update(proposal_update)

            with torch.no_grad():
                probs_prime = actor.forward(states, clamp=True)

                L_prime, KLD_prime = actor.compute_trpo_loss(probs_prime, probs, actions, advantages)

            delta_L = L_prime - L

            if delta_L > 0 and KLD_prime <= max_d_kl:
                break

            else:
                actor.apply_update(-proposal_update)

        mean_total_rewards = np.mean(rollout_total_rewards)
        se_total_rewards = np.std(rollout_total_rewards) / np.sqrt(batch_size)

        total_rewards_mean.append(mean_total_rewards)
        total_rewards_se.append(se_total_rewards)

        print(f'{epoch:>3d} | Reward {mean_total_rewards:>7.2f} +/- {se_total_rewards:<5.2f}, Max step length {max_length:.2f},  Step norm {np.linalg.norm(proposal_update):.2e}')

    total_rewards_mean = np.array(total_rewards_mean)
    total_rewards_se = np.array(total_rewards_se)

    plt.plot(np.arange(epochs), total_rewards_mean, 'k-')
    plt.fill_between(np.arange(epochs),
                     total_rewards_mean - total_rewards_se,
                     total_rewards_mean + total_rewards_se,
                     alpha=0.5)
    plt.show()

# Train our agent
train(50)
