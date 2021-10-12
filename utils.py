import os
import re
import random

import numpy as np
import torch


def find_version(experiment_version: str, checkpoint_dir: str, debug: bool = False):

    version = 0

    if debug:
        version = 'debug'
    else:
        for subdir, dirs, files in os.walk(f"{checkpoint_dir}/{experiment_version}"):
            match = re.search(r".*version_([0-9]+)$", subdir)
            if match:
                match_version = int(match.group(1))
                if match_version > version:
                    version = match_version

        version = str(version + 1)

    full_version = experiment_version + "/version_" + str(version)

    return full_version, experiment_version, "version_" + str(version)


def set_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    env.seed(seed)

def set_deterministic():
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
