import os
from typing import Optional, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

from prepare_results_drl import collect_simulation_data


def drl_plot_error(sim_ids: Iterable[int],
                   data_path: Optional[str] = 'analysis/',
                   agent_names: tuple = ('ppo', 'td3', 'sac'),
                   save_path: Optional[str] = 'analysis/drl_error_plot.pdf',
                   add_steps: bool = False,
                   add_value_labels: bool = False) -> None:

    # create axis for plotting
    num_rows = 3 if add_steps else 2
    fig, axs = plt.subplots(nrows=num_rows, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0)

    color_schemes = ('blue', 'orange', 'green')

    for agent_name, color in zip(agent_names, color_schemes):
        parquet_file = f'{data_path}{agent_name}_simulation_results.parquet'

        if not os.path.isfile(parquet_file):
            df = collect_simulation_data(agent_name, sim_ids, data_path)
        else:
            df = pd.read_parquet(parquet_file)

        # Calculate mean error and standard error for each n_training_trials, aggregated across all simulations
        error_stats = df.groupby('n_training_trials')['error'].agg(['mean', 'std']).reset_index()
        error_stats.columns = ['n_training_trials', 'mean_error', 'std']
        error_stats['error_se'] = error_stats['std'] / np.sqrt(error_stats['n_training_trials'].count())

        # Calculate mean reward and standard error for each n_training_trials, aggregated across all simulations
        reward_stats = df.groupby('n_training_trials')['total_reward'].agg(['mean', 'std']).reset_index()
        reward_stats.columns = ['n_training_trials', 'mean_reward', 'std']
        reward_stats['reward_se'] = reward_stats['std'] / np.sqrt(reward_stats['n_training_trials'].count())
        error_stats = error_stats.merge(reward_stats, on='n_training_trials')

        # Calculate mean steps and standard error for each n_training_trials, aggregated across all simulations
        if add_steps:
            steps_stats = df.groupby('n_training_trials')['steps'].agg(['mean', 'std']).reset_index()
            steps_stats.columns = ['n_training_trials', 'mean_steps', 'std']
            steps_stats['step_se'] = steps_stats['std'] / np.sqrt(steps_stats['n_training_trials'].count())
            error_stats = error_stats.merge(steps_stats, on='n_training_trials')

        # Calculate the upper and lower bounds for the error range
        error_stats['lower_bound'] = error_stats['mean_error'] - error_stats['error_se']
        error_stats['upper_bound'] = error_stats['mean_error'] + error_stats['error_se']

        # Create the plot
        # normalize training trials
        error_stats['n_training_trials'] /= 1000

        # Plot the mean line
        axs[0].plot(error_stats['n_training_trials'], error_stats['mean_error'], c=color, label=f'{agent_name.upper()} Mean Error')

        # Add the shaded error range
        axs[0].fill_between(error_stats['n_training_trials'], error_stats['lower_bound'], error_stats['upper_bound'],
                            color=color, alpha=0.3)

        axs[0].set_xticks(error_stats['n_training_trials'].unique())

        # Add value labels
        if add_value_labels:
            for _, row in error_stats.iterrows():
                axs[0].text(row['n_training_trials']+np.random.uniform(-1, 1), row['mean_error'],
                         f'{row["mean_error"]:.2f}±{row["error_se"]:.2f}',
                         ha='right', va='bottom', fontsize=8, c=color)

        axs[1].plot(error_stats['n_training_trials'], error_stats['mean_reward'], color, label=f'{agent_name.upper()} Mean Reward')
        axs[1].fill_between(error_stats['n_training_trials'], error_stats['mean_reward'] - error_stats['reward_se'],
                            error_stats['mean_reward'] + error_stats['reward_se'], color=color,
                            alpha=0.3)

        if not add_steps:
            axs[1].set_xticks(error_stats['n_training_trials'].unique(), np.uint8(error_stats['n_training_trials'].unique()), fontsize=16)

        if add_steps:
            axs[2].plot(error_stats['n_training_trials'], error_stats['mean_steps'], color, label=f'{agent_name.upper()} Mean Steps')
            axs[2].fill_between(error_stats['n_training_trials'], error_stats['mean_steps'] - error_stats['step_se'],
                                error_stats['mean_steps'] + error_stats['step_se'], color=color,
                                alpha=0.3)


        plt.tight_layout()
        for ax in axs:
            ax.grid()

        # add axis labels
        # error plot
        axs[0].set_yticks(np.arange(0, 201, 25), np.arange(0, 201, 25), fontsize=16)
        axs[0].set_ylabel('Mean Error in [mm]', fontsize=16)
        axs[0].set_xlim(1, 128)
        axs[0].legend(fontsize=16)

        # reward plot
        axs[1].set_ylabel('Mean Reward', fontsize=16)
        axs[1].set_yticks(np.arange(-400, 50, 100), np.arange(-400, 50, 100, dtype=np.int32), fontsize=16)
        axs[1].set_ylim(-400, 50)
        axs[1].set_xlim(1,
                        128)
        axs[1].legend(fontsize=16, loc="lower right")
        axs[1].set_xlabel('N training trials in thousands', fontsize=16)
        axs[1].set_xticks((1, 2, 4, 8, 16, 32, 64, 128),
                          (None, 2, 4, 8, 16, 32, 64, 128), fontsize=14)

        # steps plot
        if add_steps:
            axs[2].set_xticks(error_stats['n_training_trials'].unique(), np.uint8(error_stats['n_training_trials'].unique()),
                              fontsize=16)
            axs[2].set_xlabel('N training trials in thousands', fontsize=16)
            axs[2].set_ylabel('Mean Steps', fontsize=16)
            axs[2].set_xlim(np.min(error_stats['n_training_trials'].unique()),
                            np.max(error_stats['n_training_trials'].unique()))
            axs[2].legend(fontsize=16)

        # Save the plot
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()


if __name__ == '__main__':
    sim_ids = (1, 2, 3, 4, 5)
    drl_plot_error(sim_ids, add_steps=True)
