class HyperParams:
    gamma = 0.99
    lamda = 0.98
    #hidden = 64
    hidden = 30
    #critic_lr = 0.0003
    critic_lr = 0.01
    #actor_lr = 0.0003
    actor_lr = 0.01
    #batch_size = 64
    batch_size = 30 # For swimmer
    #batch_size = 50 # For walker
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2
