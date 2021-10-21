# UvA_RL_2021_Reproducibility_Lab
===================
### General info ###
This folder contains all the necessary files for Reproducible research assignment.


===================
### Folders included ###
- continuous_trpo            : contains all files related to continuous TRPO training.
- discrete_trpo              : contains all files related to discrete TRPO training.
- Figures                    : TRPO training figures. 


### Generating results ###

### 1. Setting up conda environment.
Create conda environment from .yml:
```
conda env create -f environment.yml
```

Start the environment
```
conda activate rlcourse
```


### 2. Running the continuous TRPO files.

Install Mujoco
```
# TODO
```


Navigate to the evoman_framework folder
```
cd continuous_trpo/mujoco
```

Run the experiment
```
python main.py --env [environment]
```
The results are stored in the folder: [logs_json]/

environment can be any of [Swimmer-v2, Walker2d-v2, Hopper-v2].

