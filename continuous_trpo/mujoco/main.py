"""
Trains a (continuous) PG method
Original code from: https://reinforcement-learning-kr.github.io/2018/06/24/5_trpo/
"""
import os
import gym
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils.utils import get_action, save_checkpoint, stats_to_writer, format_tf
from collections import deque
from utils.running_state import ZFilter
from hparams import HyperParams as hp
from tensorboardX import SummaryWriter
from collections import defaultdict
from pathlib import Path
import json
import time
import shutil
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='TRPO',
                    help='select one of algorithms among Vanilla_PG,'
                         'NPG, TPRO, PPO')
parser.add_argument('--env', type=str, default="Swimmer-v2",
                    help='name of Mujoco environement')

parser.add_argument('--logging_freq', type=int, default=100,
                    help='How frequent to log stats.')
parser.add_argument('--plot_freq', type=int, default=1,
                    help='How frequent to log to tensorboard.')
parser.add_argument('--num_iters', type=int, default=500,
                    help='Number of iterations')
parser.add_argument('--num_sessions', type=int, default=10,
                    help='Number of sessions to average the results over')

parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

if args.algorithm == "PG":
    from agent.vanila_pg import train_model
elif args.algorithm == "NPG":
    from agent.tnpg import train_model
elif args.algorithm == "TRPO":
    from agent.trpo_gae import train_model
elif args.algorithm == "PPO":
    from agent.ppo_gae import train_model
else:
    raise "Unknown method"


if __name__=="__main__":
    # you can choose other environments.
    # possible environments: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2,
    # HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2,
    # Walker2d-v2
    logging_freq = args.logging_freq
    num_iters = args.num_iters
    plot_freq = args.plot_freq
    num_sessions = args.num_sessions
    env = gym.make(args.env)

    # Create an empty output direction
    output_dir = "./logs_json/{}".format(args.env)
    shutil.rmtree(output_dir, ignore_errors=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Start training on env {} with algorithm {}".format(args.env, args.algorithm))

    for session in range(num_sessions + 1):
        full_stats = defaultdict(list)
        env.seed(500 + session)
        torch.manual_seed(500 + session)

        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        # writer = SummaryWriter(args.logdir + "/" + args.env + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(args.logdir + "/" + args.env + "/" + str(session))
        actor = Actor(num_inputs, num_actions)
        critic = Critic(num_inputs)

        running_state = ZFilter((num_inputs,), clip=5)

        if args.load_model is not None:
            saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
            ckpt = torch.load(saved_ckpt_path)

            actor.load_state_dict(ckpt['actor'])
            critic.load_state_dict(ckpt['critic'])

            running_state.rs.n = ckpt['z_filter_n']
            running_state.rs.mean = ckpt['z_filter_m']
            running_state.rs.sum_square = ckpt['z_filter_s']
            print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

        actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
        critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                                weight_decay=hp.l2_rate)

        episodes = 0
        for iter in range(num_iters):
            actor.eval(), critic.eval()
            memory = deque()

            steps = 0
            scores = []

            while steps < 2048:
                
                episodes += 1
                state = env.reset()
                state = running_state(state)
                score = 0
                for _ in range(10000):
                    if args.render:
                        env.render()

                    steps += 1
                    mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                    action = get_action(mu, std)[0]

                    next_state, reward, done, _ = env.step(action)
                    next_state = running_state(next_state)

                    if done:
                        mask = 0
                    else:
                        mask = 1

                    memory.append([state, action, reward, mask])

                    score += reward
                    state = next_state

                    if done:
                        break
                
                scores.append(score)

            score_avg = np.mean(scores)
            print('session {} iter {} episode score  {:.2f}'.format(session, iter, score_avg))

            actor.train(), critic.train()

            if args.algorithm != "TRPO":
                train_model(actor, critic, memory, actor_optim, critic_optim)

            else:
                stats = train_model(actor, critic, memory, actor_optim, critic_optim)
                stats["score_avg"] = float(score_avg)

                if iter % plot_freq == 0:
                    # add stats to tensorboard
                    stats_to_writer(writer, stats, iter)

                    # Add stats to dict
                    ts = int(time.time())
                    full_stats["Mean Reward"].append(format_tf(ts, iter, stats["score_avg"]))
                    full_stats["TRPO Learning Rate/Max"].append(format_tf(ts, iter, stats["max_length"]))
                    full_stats["TRPO Learning Rate/Line Search Scalar"].append(format_tf(ts, iter, stats["KL_boundary_coeff"]))
                    full_stats["TRPO Learning Rate/Effective Learning rate"].append(format_tf(ts, iter, stats["Effective Learning rate"]))
                    full_stats["Update/Loss Improvement"].append(format_tf(ts, iter, stats["delta_L"]))
                    full_stats["Update/KL Divergence"].append(format_tf(ts, iter, stats["KLD_prime"]))
                    full_stats["Norms/grad"].append(format_tf(ts, iter, float(np.linalg.norm(stats["norms_grad"]))))
                    full_stats["Norms/update"].append(format_tf(ts, iter, float(np.linalg.norm(stats["full_step"]))))
                    full_stats["Norms/search_dir"].append(format_tf(ts, iter, float(np.linalg.norm(stats["search_dir"]))))
     
            if iter % logging_freq:
                score_avg = int(score_avg)

                model_path = os.path.join(os.getcwd(),'save_model')
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)

                ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

                save_checkpoint({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'z_filter_n':running_state.rs.n,
                    'z_filter_m': running_state.rs.mean,
                    'z_filter_s': running_state.rs.sum_square,
                    'args': args,
                    'score': score_avg
                }, filename=ckpt_path)

        for key, value in full_stats.items():
            keyname = key.replace('/', '_').replace(' ', '_')
            filename = "/run-{}_{}.json".format(session, keyname)
            print("dumping ", filename)
            with open(output_dir + filename, 'w') as fp:
                json.dump(value, fp)

