import numpy as np
import os
from glob import glob
from run_reaching_ppo import ReachingEnvironment, PPOAgent, render_reaching


if __name__ == '__main__':
    load_path: str = 'results/training_ppo_1/model_*/'
    for file_path in glob(load_path):
        # Extract n_training_trials from the file path
        folder_name = os.path.basename(os.path.dirname(file_path))
        n_training_trials = int(folder_name.split('_')[1])

        # initialize agent
        state_dim = 6  # Current joint angles (2) + cartesian error to target position (2)
        action_dim = 2  # Changes in joint angles
        agent = PPOAgent(input_dim=state_dim, output_dim=action_dim)
        agent.load(file_path)

        render_reaching(PPOAgent=agent,
                        init_thetas=np.radians((90, 90)),
                        max_steps=100,
                        fps=200,
                        save_path=f'analysis/ppo_ntrain[{n_training_trials}]_animation.mp4')