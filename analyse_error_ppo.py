import os
from typing import Optional, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

from prepare_results_ppo import ppo_collect_simulation_data


def ppo_plot_error(sim_ids: Iterable[int],
                   data_path: Optional[str] = 'analysis/ppo_simulation_results.parquet',
                   save_path: Optional[str] = 'analysis/ppo_error_plot.pdf') -> None:

    if not os.path.isfile(data_path):
        df = ppo_collect_simulation_data(sim_ids, data_path)
    else:
        df = pd.read_parquet(data_path)

    # Calculate mean error and standard error for each n_training_trials, aggregated across all simulations
    error_stats = df.groupby('n_training_trials')['error'].agg(['mean', 'sem']).reset_index()
    error_stats.columns = ['n_training_trials', 'mean_error', 'sem_error']

    # Calculate mean reward and standard error for each n_training_trials, aggregated across all simulations
    reward_stats = df.groupby('n_training_trials')['total_reward'].agg(['mean', 'sem']).reset_index()
    reward_stats.columns = ['n_training_trials', 'mean_reward', 'sem_reward']
    error_stats = error_stats.merge(reward_stats, on='n_training_trials')

    # Calculate the upper and lower bounds for the error range
    error_stats['lower_bound'] = error_stats['mean_error'] - error_stats['sem_error']
    error_stats['upper_bound'] = error_stats['mean_error'] + error_stats['sem_error']

    # Create the plot
    fig, axs = plt.subplots(nrows=2, figsize=(10, 6))

    # Plot the mean line
    axs[0].plot(error_stats['n_training_trials'], error_stats['mean_error'], 'b-', label='Mean Error')

    # Add the shaded error range
    axs[0].fill_between(error_stats['n_training_trials'], error_stats['lower_bound'], error_stats['upper_bound'],
                     alpha=0.3, label='Standard Error Range')
    axs[0].set_xticks(df['n_training_trials'].unique())
    axs[0].set_xlabel('Number of Training Trials')
    axs[0].set_ylabel('Mean Error in [mm]')
    axs[0].legend()

    # Add value labels
    for _, row in error_stats.iterrows():
        axs[0].text(row['n_training_trials'], row['mean_error'],
                 f'{row["mean_error"]:.2f}±{row["sem_error"]:.2f}',
                 ha='right', va='bottom')

    axs[1].plot(error_stats['n_training_trials'], error_stats['mean_reward'], 'r-', label='Mean Reward')
    axs[1].fill_between(error_stats['n_training_trials'], error_stats['mean_reward'] - error_stats['sem_reward'],
                        error_stats['mean_reward'] + error_stats['sem_reward'], color='r',
                        alpha=0.3, label='Standard Error Range')
    axs[1].set_xticks(df['n_training_trials'].unique())
    axs[1].set_xlabel('Number of Training Trials')
    axs[1].set_ylabel('Mean Reward')
    axs[1].legend()

    plt.tight_layout()

    # Save the plot
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()


if __name__ == '__main__':
    sim_ids = (1, 2, 3)
    ppo_plot_error(sim_ids)
