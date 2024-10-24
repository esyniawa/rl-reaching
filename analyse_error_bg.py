import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Iterable

from prepare_results_bg import collect_simulation_data, collect_perturbation_data


def plot_error(data_path: Optional[str] = 'analysis/bg_simulation_results.parquet',
               save_path: Optional[str] = 'analysis/bg_error_plot.pdf',
               error_column: str = 'error',
               add_value_labels: bool = False) -> None:

    df = pd.read_parquet(data_path)

    # Calculate mean error and standard error for each n_training_trials, aggregated across all simulations
    error_stats = df.groupby('n_training_trials')[error_column].agg(['mean', 'std']).reset_index()
    error_stats.columns = ['n_training_trials', 'mean_error', 'std']
    error_stats['se'] = error_stats['std'] / np.sqrt(error_stats['n_training_trials'].count())

    # Calculate the upper and lower bounds for the error range
    error_stats['lower_bound'] = error_stats['mean_error'] - error_stats['se']
    error_stats['upper_bound'] = error_stats['mean_error'] + error_stats['se']

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the mean line
    plt.plot(error_stats['n_training_trials']/1000, error_stats['mean_error'], 'b-', label='Mean Error')

    # Add the shaded error range
    plt.fill_between(error_stats['n_training_trials']/1000, error_stats['lower_bound'], error_stats['upper_bound'],
                     alpha=0.3, label='Standard Error')
    plt.xticks((1, 2, 4, 8, 16, 32), fontsize=14)
    plt.yticks(np.arange(0, 140, 10), fontsize=14)
    plt.xlim(1, 32)
    plt.xlabel('N training trials in thousands', fontsize=18)
    plt.ylabel('Mean Error in [mm]', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)

    # Add value labels
    if add_value_labels:
        for _, row in error_stats.iterrows():
            plt.text(row['n_training_trials'], row['mean_error'],
                     f'{row["mean_error"]:.2f}±{row["se"]:.2f}',
                     ha='right', va='bottom')

    plt.tight_layout()

    # Save the plot
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()


if __name__ == '__main__':

    sim_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    df_reaching = collect_simulation_data(sim_ids)
    df_perturbation = collect_perturbation_data(sim_ids)

    # plot error in reaching task
    plot_error(data_path='analysis/bg_simulation_results.parquet',
               save_path='analysis/bg_error_plot.pdf',
               error_column='error')

    # plot error in reaching task with perturbation
    plot_error(data_path='analysis/bg_simulation_results_perturbation.parquet',
               save_path='analysis/bg_error_plot_perturbation.pdf',
               error_column='error_after_pert')
