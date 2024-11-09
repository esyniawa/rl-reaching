import numpy as np
import os
import argparse
from glob import glob


if __name__ == '__main__':
    sim_args_parser = argparse.ArgumentParser()
    sim_args_parser.add_argument('--id', type=int, default=0, help='Simulation ID')
    sim_args_parser.add_argument('--fps', type=int, default=20)
    sim_args_parser.add_argument('--drl_model', type=str, default='ppo', choices=['ppo', 'td3', 'sac'])
    sim_args = sim_args_parser.parse_args()

    simID: int = sim_args.id
    if sim_args.drl_model == 'ppo':
        from run_reaching_ppo import *
    elif sim_args.drl_model == 'td3':
        from run_reaching_td3 import *
    elif sim_args.drl_model == 'sac':
        from run_reaching_sac import *
    else:
        raise NotImplementedError(f"Network name {sim_args.drl_model} is not supported.")

    load_path: str = f'results/training_{sim_args.drl_model}_{simID}/model_*/'
    for file_path in glob(load_path):
        # Extract n_training_trials from the file path
        folder_name = os.path.basename(os.path.dirname(file_path))
        n_training_trials = int(folder_name.split('_')[1])

        # initialize agent
        state_dim = 6  # Current joint angles (2) + cartesian error to target position (2)
        action_dim = 2  # Changes in joint angles
        if sim_args.drl_model == 'ppo':
            agent = PPOAgent(input_dim=state_dim, output_dim=action_dim)
        elif sim_args.drl_model == 'td3':
            agent = TD3Agent(input_dim=state_dim, output_dim=action_dim)
        elif sim_args.drl_model == 'sac':
            agent = SACAgent(input_dim=state_dim, output_dim=action_dim)
        agent.load(file_path)

        render_reaching(Agent=agent,
                        init_thetas=np.radians((90, 90)),
                        max_steps=500,
                        fps=20,
                        save_path=f'analysis/{sim_args.drl_model}_ntrain[{n_training_trials}]_animation.mp4')