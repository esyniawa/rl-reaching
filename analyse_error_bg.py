import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Iterable

from prepare_results_bg import collect_simulation_data


def plot_error(sim_ids: Iterable[int],
               data_path: Optional[str] = 'analysis/bg_simulation_results.parquet',
               save_path: Optional[str] = 'analysis/bg_error_plot.pdf') -> None:

    if not os.path.isfile(data_path):
        df = collect_simulation_data(sim_ids, data_path)
    else:
        df = pd.read_parquet(data_path)

    # Calculate mean error and standard error for each n_training_trials, aggregated across all simulations
    error_stats = df.groupby('n_training_trials')['error'].agg(['mean', 'sem']).reset_index()
    error_stats.columns = ['n_training_trials', 'mean_error', 'sem_error']

    # Calculate the upper and lower bounds for the error range
    error_stats['lower_bound'] = error_stats['mean_error'] - error_stats['sem_error']
    error_stats['upper_bound'] = error_stats['mean_error'] + error_stats['sem_error']

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the mean line
    plt.plot(error_stats['n_training_trials'], error_stats['mean_error'], 'b-', label='Mean Error')

    # Add the shaded error range
    plt.fill_between(error_stats['n_training_trials'], error_stats['lower_bound'], error_stats['upper_bound'],
                     alpha=0.3, label='Standard Error Range')
    plt.xticks(df['n_training_trials'].unique())
    plt.xlabel('Number of Training Trials')
    plt.ylabel('Mean Error in [mm]')
    plt.legend()

    # Add value labels
    for _, row in error_stats.iterrows():
        plt.text(row['n_training_trials'], row['mean_error'],
                 f'{row["mean_error"]:.2f}±{row["sem_error"]:.2f}',
                 ha='right', va='bottom')

    plt.tight_layout()

    # Save the plot
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()


if __name__ == '__main__':

    sim_ids = (1, 2, 3, 4)
    plot_error(sim_ids)
