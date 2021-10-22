import os
import argparse
import warnings
import yaml
from collections import namedtuple
from shutil import copyfile

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import gym

from discrete_trpo.models import ActorDiscrete, Critic
from discrete_trpo.rl_utils import estimate_advantages, flat_grad, conjugate_gradient
from discrete_trpo.utils import find_version, set_seed, set_deterministic

CHECKPOINT_DIR = './discrete_trpo/checkpoints'

# ==============================================================================
# Main loop
# ==============================================================================
def train(args, seed=None):
    # ==========================================================================
    # Read in config file
    # ==========================================================================
    with open(args.config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    print(50 * "+")
    print("HYPER-PARAMETERS")
    print(yaml.dump(config))
    print(50 * "+")

    if seed == None:
        seed = config['run']['seed']
    else:
        print(f"\nExperiment seed overwritten to {seed}!\n")

    # ==========================================================================
    # Experimental Setup
    # ==========================================================================
    print("\nEXPERIMENT SETUP")

    # == Version
    # ==== ./checkpoints/data_version/version_number
    experiment_dir = config['env'] + \
        "_" + config['run']['experiment_name']

    if config['run']['debug']:
        full_version = f"{experiment_dir}/debug"
    else:
        full_version = f"{experiment_dir}/seed_{seed}"

    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}", exist_ok=True)
    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}/checkpoints", exist_ok=True)

    copyfile(args.config_file_path,
             f"{CHECKPOINT_DIR}/{full_version}/{os.path.split(args.config_file_path)[-1]}")

    # == Device
    # Probably not needed
    # TODO: implement GPU training
    use_cuda = False #config['run']['gpu'] or config['run']['gpu'] > 1
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Training on {device}")

    # == Logging
    writer = SummaryWriter(log_dir=f"{CHECKPOINT_DIR}/{full_version}")

    print(f"Saving to {CHECKPOINT_DIR}/{full_version}/checkpoints")

    # == Environment Setup
    env = gym.make(config['env'])

    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # == Reproducibility
    set_seed(seed, env)
    if config['run']['set_gpu_deterministic']:
        set_deterministic()

    # ==========================================================================
    # TRAINING
    # ==========================================================================
    print("\nTRAINING")
    #TODO: add functionality for other actors, e.g. GaussianMLP
    actor = ActorDiscrete(num_inputs=state_size,
                          num_actions=num_actions,
                          hidden=config['actor']['hidden'])

    critic = Critic(num_inputs=state_size,
                    hidden=config['critic']['hidden'])
    critic_optimizer = Adam(critic.parameters(),
                            lr=config['critic']['lr'])

    Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', ])

    total_rewards_mean = []
    total_rewards_se = []

    for episode in range(config['train']['episodes']):
        rollouts = []
        rollout_total_rewards = []

        # ======================================================================
        # Generate batched episode
        # ======================================================================
        actor.eval()
        critic.eval()

        for _ in range(config['train']['batch_size']):
            state = env.reset()
            done = False

            samples = []

            i = 0
            while not done:
                with torch.no_grad():
                    action = actor.sample_action(state)

                next_state, reward, done, _ = env.step(action)

                samples.append((state, action, reward, next_state))

                state = next_state

                if i > config['train']['max_steps_per_sample']:
                    break

                i += 1

            # Transpose samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state)
                                 for state in states], dim=0).float()
            next_states = torch.stack(
                [torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))

            rollout_total_rewards.append(rewards.sum().item())

        # ======================================================================
        # Use episode to gather states, actions and advantages based on both
        # ======================================================================
        actor.train()
        critic.train()

        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

        advantages = estimate_advantages(rollouts, critic)

        # ======================================================================
        # TRPO Actor Loss
        # ======================================================================
        probs = actor.forward(states, clamp=True)

        L, KLD = actor.compute_trpo_loss(probs, probs, actions, advantages)

        pre_update_policy_entropy = actor.policy_entropy(probs.detach())

        # ======================================================================
        # Gather gradients & KL + Fisher matrix for actor update
        # ======================================================================
        parameters = list(actor.parameters())

        g = flat_grad(L, parameters, retain_graph=True)
        d_kl = flat_grad(KLD, parameters, create_graph=True)

        def HVP(v):
            return flat_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * config['trpo']['max_d_kl'] / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        # Clip gradient
        if episode / config['train']['episodes'] >= 0.01:
            if np.linalg.norm(max_step) >= config['trpo']['max_grad_norm']:
                clip_coeff = config['trpo']['max_grad_norm'] / (np.linalg.norm(max_step) + 1e-6)

                if clip_coeff < 1:
                    max_step = clip_coeff * max_step

            if np.linalg.norm(max_step) > config['trpo']['max_grad_norm']:
                print(f"Update norm, {np.linalg.norm(max_step):.2f}, out-of-bounds. Skipping.")
                continue

        # ======================================================================
        # Line search over possible step sizes
        # ======================================================================
        for ii in range(config['trpo']['max_ls_steps']):

            KL_boundary_coeff = config['trpo']['ls_step_coef'] ** ii
            proposal_update =  KL_boundary_coeff * max_step

            actor.apply_update(proposal_update)

            with torch.no_grad():
                probs_prime = actor.forward(states, clamp=True)

                post_update_policy_entropy = actor.policy_entropy(probs_prime.detach())

                L_prime, KLD_prime = actor.compute_trpo_loss(
                    probs_prime, probs, actions, advantages)

            delta_L = L_prime - L

            if delta_L > 0 and KLD_prime <= config['trpo']['max_d_kl']:
                break

            #elif ii != config['trpo']['max_ls_steps'] - 1:
            actor.apply_update(-proposal_update)

        mean_total_rewards = np.mean(rollout_total_rewards)
        se_total_rewards = np.std(rollout_total_rewards) / np.sqrt(config['train']['batch_size'])

        total_rewards_mean.append(mean_total_rewards)
        total_rewards_se.append(se_total_rewards)

        # ======================================================================
        # Update the critic
        # ======================================================================
        critic_loss = 0.5 * torch.mean(advantages ** 2)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ======================================================================
        # Logging
        # ======================================================================
        if episode % config['run']['logging_frequency'] == 0:
            print(f'{episode:>3d} | Reward {mean_total_rewards:>7.2f} +/- {se_total_rewards:<5.2f}, Max step length {max_length:>5.2f},  KL-boundary coeff {KL_boundary_coeff :.2f}, Effective learning rate {KL_boundary_coeff * max_length :>5.2f},  Step norm {np.linalg.norm(proposal_update):.2e}')

            writer.add_scalar(
                tag='Mean Reward',
                scalar_value=mean_total_rewards,
                global_step=episode
            )

            writer.add_scalar(
                tag='TRPO Learning Rate/Max',
                scalar_value=max_length,
                global_step=episode
            )

            writer.add_scalar(
                tag='TRPO Learning Rate/Line Search Scalar',
                scalar_value=KL_boundary_coeff,
                global_step=episode
            )

            writer.add_scalar(
                tag='TRPO Learning Rate/Effective Learning Rate',
                scalar_value=KL_boundary_coeff * max_length,
                global_step=episode
            )

            writer.add_scalar(
                tag='Policy Mean Entropy/Pre Update',
                scalar_value=pre_update_policy_entropy,
                global_step=episode
            )

            writer.add_scalar(
                tag='Policy Mean Entropy/Post Update',
                scalar_value=post_update_policy_entropy,
                global_step=episode
            )

            writer.add_scalar(
                tag='Update/Loss Improvement',
                scalar_value=delta_L,
                global_step=episode
            )

            writer.add_scalar(
                tag='Update/KL Divergence',
                scalar_value=KLD_prime,
                global_step=episode
            )

            writer.add_scalar(
                tag='Norms/Grad',
                scalar_value=np.linalg.norm(g),
                global_step=episode
            )

            writer.add_scalar(
                tag='Norms/SearchDir',
                scalar_value=np.linalg.norm(search_dir),
                global_step=episode
            )

            writer.add_scalar(
                tag='Norms/Update',
                scalar_value=np.linalg.norm(proposal_update),
                global_step=episode
            )

            writer.flush()

    total_rewards_mean = np.array(total_rewards_mean)
    total_rewards_se = np.array(total_rewards_se)

    if config['run']['plot_rewards']:
        plt.plot(np.arange(config['train']['episodes']), total_rewards_mean, 'k-')
        plt.fill_between(np.arange(config['train']['episodes']),
                         total_rewards_mean - total_rewards_se,
                         total_rewards_mean + total_rewards_se,
                         alpha=0.5)
        plt.show()

    writer.close()

    torch.save({
            'episode': episode,
            'reward': total_rewards_mean[-1],
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict()
        },
               f"{CHECKPOINT_DIR}/{full_version}/checkpoints/checkpoint")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters
    parser.add_argument('--config_file_path',
                        default='./discrete_trpo/configs/CartPole_v1.yaml',
                        type=str)

    parser.add_argument('--seed',
                        default=[0,1,2,3,4,5,6,7,8,9],
                        type=int,
                        nargs='+')

    args = parser.parse_args()

    #* WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS
    warnings.filterwarnings('ignore', message=r'.*Named tensors.*')
    warnings.filterwarnings('ignore', message=r'.*Matplotlib*')

    if isinstance(args.seed, list):
        # If multiple seeds are provided
        for seed in args.seed:
            train(args, seed=seed)

    elif isinstance(args.seed, int):
        # If only 1
        train(args, seed=args.seed)

    else:
        # None
        # Uses default in yaml file
        train(args)
