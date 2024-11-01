import numpy as np
import os
import sys
from glob import glob
from run_reaching_td3 import *


if __name__ == '__main__':
    simID: int = int(sys.argv[1])

    load_path: str = f'results/training_td3_{simID}/model_*/'
    for file_path in glob(load_path):
        # Extract n_training_trials from the file path
        folder_name = os.path.basename(os.path.dirname(file_path))
        n_training_trials = int(folder_name.split('_')[1])

        # initialize agent
        state_dim = 6  # Current joint angles (2) + cartesian error to target position (2)
        action_dim = 2  # Changes in joint angles
        agent = TD3Agent(input_dim=state_dim, output_dim=action_dim)
        agent.load(file_path)

        render_reaching(Agent=agent,
                        init_thetas=np.radians((90, 90)),
                        max_steps=500,
                        fps=20,
                        save_path=f'analysis/TD3_ntrain[{n_training_trials}]_animation.mp4')