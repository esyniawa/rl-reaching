import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Optional, Iterable


def collect_simulation_data(sim_ids: Iterable[int],
                            save_path: Optional[str] = 'analysis/bg_simulation_results.parquet') -> pd.DataFrame:

    all_data = []

    for sim_id in sim_ids:
        pattern = f'results/test_run_model_{sim_id}/model_*/data.npz'
        for file_path in glob(pattern):
            # Extract n_training_trials from the file path
            folder_name = os.path.basename(os.path.dirname(file_path))
            n_training_trials = int(folder_name.split('_')[1])

            # Load the data
            data = np.load(file_path)

            # Determine the number of trials
            num_test_trials = data['error'].shape[0]

            # Add all keys from the npz file to the dictionary
            for test_trial in range(num_test_trials):
                # Create a dictionary with sim_id and n_training_trials
                row_data = {
                    'sim_id': sim_id,
                    'n_training_trials': n_training_trials,
                    'test_trial': test_trial,
                }

                for key in data.keys():
                   row_data[key] = data[key][test_trial]

                # Append the dictionary to the list
                all_data.append(row_data)

    # Create DataFrame from all collected data
    df = pd.DataFrame(all_data)

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        df.to_parquet(save_path)

    return df


def collect_perturbation_data(sim_ids: Iterable[int],
                            save_path: Optional[str] = 'analysis/bg_simulation_results_perturbation.parquet') -> pd.DataFrame:

    all_data = []

    for sim_id in sim_ids:
        pattern = f'results/test_pert_run_model_{sim_id}/model_*/data.npz'
        for file_path in glob(pattern):
            # Extract n_training_trials from the file path
            folder_name = os.path.basename(os.path.dirname(file_path))
            n_training_trials = int(folder_name.split('_')[1])

            # Load the data
            data = np.load(file_path)

            # Determine the number of trials
            num_test_trials = data['error_after_pert'].shape[0]

            # Add all keys from the npz file to the dictionary
            for test_trial in range(num_test_trials):
                # Create a dictionary with sim_id and n_training_trials
                row_data = {
                    'sim_id': sim_id,
                    'n_training_trials': n_training_trials,
                    'test_trial': test_trial,
                }

                for key in data.keys():
                   row_data[key] = data[key][test_trial]

                # Append the dictionary to the list
                all_data.append(row_data)

    # Create DataFrame from all collected data
    df = pd.DataFrame(all_data)

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        df.to_parquet(save_path)

    return df


if __name__ == '__main__':

    debug: bool = True

    sim_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    result_df = collect_simulation_data(sim_ids)

    # Display the first few rows of the DataFrame
    print(result_df.head())
    print(result_df.columns)

    result_pert_df = collect_simulation_data(sim_ids, save_path='analysis/bg_simulation_results_perturbation.parquet')

    # Display the first few rows of the DataFrame
    print(result_pert_df.head())
    print(result_pert_df.columns)