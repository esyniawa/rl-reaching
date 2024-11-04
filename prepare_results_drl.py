import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Optional, Iterable


def collect_simulation_data(
        network_name: str,
        sim_ids: Iterable[int],
        save_path: Optional[str] = 'analysis/') -> pd.DataFrame:

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = f'{network_name}_simulation_results.parquet'

    if network_name not in ['ppo', 'td3', 'sac']:
        raise NotImplementedError(f"Network name {network_name} is not supported.")

    all_data = []

    for sim_id in sim_ids:
        pattern = f'results/test_{network_name}_{sim_id}/model_*/results.npz'
        for file_path in glob(pattern):
            # Extract n_training_trials from the file path
            folder_name = os.path.basename(os.path.dirname(file_path))
            n_training_trials = int(folder_name.split('_')[1])

            # Load the data
            data = np.load(file_path)

            # Determine the number of trials
            num_test_trials = data['total_reward'].shape[0]

            # Add all keys from the npz file to the dictionary
            for test_trial in range(num_test_trials):
                # Create a dictionary with sim_id and n_training_trials
                row_data = {
                    'sim_id': sim_id,
                    'n_training_trials': n_training_trials,
                    'test_trial': test_trial,
                }
                for key in data.keys():
                    if not key == 'success_rate':
                        row_data[key] = data[key][test_trial]

                # Append the dictionary to the list
                all_data.append(row_data)

    # Create DataFrame from all collected data
    df = pd.DataFrame(all_data)

    if save_path is not None:
        df.to_parquet(save_path + save_name, index=False)

    return df


if __name__ == '__main__':
    debug: bool = True

    names = ['ppo', 'td3', 'sac']
    for name in names:
        result_df = collect_simulation_data(network_name=name, sim_ids=(1, 2, 3, 4))

        if debug:
            # Display the first few rows of the DataFrame
            print(result_df.head())
            print(result_df.columns)
