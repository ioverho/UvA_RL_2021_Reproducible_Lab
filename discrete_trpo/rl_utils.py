# Blatant copy paste from:
# https://gist.github.com/elumixor/c16b7bdc38e90aa30c2825d53790d217

import torch

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
