class HyperParams:
    gamma = 0.99
    lamda = 0.98
    critic_lr = 0.01 # TRPO paper
    actor_lr = 0.01 # TRPO paper
    batch_size = 64
    hidden = 30 # for swimmer
    # hidden = 50 # for hopper and walker
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2
