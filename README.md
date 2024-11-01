# Reaching Models

This repository implements and compares different approaches for motor learning:
1. A neurocomputational model based on basal ganglia circuits
2. Deep Reinforcement Learning approaches:
   - Proximal Policy Optimization (PPO)
   - Soft Actor-Critic (SAC)
   - Twin Delayed Deep Deterministic Policy Gradient (TD3)

## Overview

The project simulates a planar arm system with two joints (shoulder and elbow) performing reaching movements. All models aim to learn and execute accurate reaching movements to target positions in the arm's workspace.

### Neurocomputational Model

The basal ganglia-based model implements:
- A biologically-inspired architecture using ANNarchy ([Artificial Neural Networks architect](https://github.com/ANNarchy/ANNarchy), Vitay et al., 2015)
- Key brain regions involved in motor control:
  - Primary Motor Cortex (M1)
  - Striatum (StrD1)
  - Substantia Nigra pars reticulata (SNr)
  - Ventrolateral Thalamus (VL)
  - Substantia Nigra pars compacta (SNc) for dopamine modulation
- Synaptic plasticity rules for learning
- Population coding for motor commands

### Deep RL Models

#### PPO Implementation
- Actor-Critic architecture with separate policy and value networks
- Gaussian action distribution for continuous action space
- Generalized Advantage Estimation (GAE)
- Experience replay buffer with mini-batch updates
- Parallel environment sampling using multiple workers

#### SAC Implementation
- Maximum entropy RL framework
- Dual Q-networks to mitigate positive bias
- Automatic entropy tuning
- Gaussian policy with reparameterization trick
- Experience replay for off-policy learning
- Soft policy updates

#### TD3 Implementation
- Twin delayed Q-learning to reduce overestimation bias
- Delayed policy updates
- Target policy smoothing
- Clipped double Q-learning
- Experience replay buffer
- Target networks with soft updates

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd [repo-name]
```

Required dependencies (see environment.yml):
- ANNarchy
- PyTorch
- NumPy
- Matplotlib
- Pandas

With Anaconda or Miniforge you can also import the environment:

```bash
# In the repository folder
conda env create -n [your-env-name] --file environment.yml 
conda activate [your-env-name]
```

## Usage

### Running the Neurocomputational Model

```bash
python run_reaching.py --id [simulation_id] [options]

Options:
  --id INT                    Simulation ID
  --test_pert BOOL           Test perturbation resistance
  --test_reach_condition STR  Testing condition ('cube', 'random', 'circle')
  --con_monitor BOOL         Enable connection monitoring
  --clean BOOL               Clean ANNarchy compilation
  --animate_arms BOOL        Enable arm animation
```

### Running Deep RL Models

```bash
# PPO
python run_reaching_ppo.py --id [simulation_id] [options]

# SAC
python run_reaching_sac.py --id [simulation_id] [options]

# TD3
python run_reaching_td3.py --id [simulation_id] [options]

Common Options:
  --id INT                   Simulation ID
  --save BOOL               Save model checkpoints
  --do_plot BOOL           Generate performance plots
  --num_workers INT         Number of parallel workers (PPO/TD3)
  --num_testing_trials INT  Number of test trials
  --buffer_size INT        Replay buffer capacity (SAC/TD3)
  --batch_size INT         Mini-batch size for updates
```

## Model Components

### Planar Arms System
- Two-joint arm system (shoulder and elbow)
- Forward and inverse kinematics
- Joint angle limits and workspace constraints
- Visualization capabilities

### Training Parameters
- Multiple training durations: 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k trials (modify them in the scipt)

## Results Directory Structure

```
results/
├── training_{algorithm}_{id}/
│   └── model_{trials}/
│       ├─── (Synaptic) weigths of the neural network 
│       └─── (Monitors)
└── test_{algorithm}_{id}/
    └── model_{trials}/
        ├── results.npz
        └── error.pdf (Plot of reaching errors in the test phase)
```

## Data Analysis

Results are saved in various formats:
- `.npz` files containing reaching performance metrics
- `.pdf` plots showing:
  - Error distribution
  - Steps to completion (for DRL Algorithms)
  - Error over trials 
  - Reward trajectories (for DRL Algorithms)

## Citation

If you use our neurocomputational model in your research, please cite:

```bibtex
@poster{proprioceptive_context_in_motor_learning,
author={Syniawa, E. and Hamker, F.H.},
title={Proprioceptive Context in Motor Learning: A Neurocomputational Study of Basal Ganglia Circuits},
year={2024},
}
```

## Acknowledgments

- [ANNarchy](https://github.com/ANNarchy/ANNarchy) development team
- [PPO algorithm authors](https://arxiv.org/abs/1707.06347) (John Schulman et al., 2017)
- [SAC algorithm authors](https://arxiv.org/abs/1812.05905) (Tuomas Haarnoja et al., 2018)
- [TD3 algorithm authors](https://proceedings.mlr.press/v80/fujimoto18a.html) (Scott Fujimoto et al., 2018)