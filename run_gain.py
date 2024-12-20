import numpy as np

from network.model import *
from monitoring import PopMonitor

from experiments import check_gain
from utils import generate_random_coordinate

import argparse
import matplotlib.pyplot as plt


def scale_array(a, low, upper):
    if len(a) == 0:
        return []

    min_val = min(a)
    max_val = max(a)

    if min_val == max_val:
        return [low] * len(a)

    return [low + (x - min_val) * (upper - low) / (max_val - min_val) for x in a]


if __name__ == '__main__':
    exp_parser = argparse.ArgumentParser()
    exp_parser.add_argument('--id', type=int, default=0, help='Simulation ID')
    exp_parser.add_argument('--display', type=bool, default=True)
    exp_parser.add_argument('--clean', type=bool, default=False, help='Clean ANNarchy compilation of the model')
    exp_args = exp_parser.parse_args()

    sim_id = exp_args.id
    sim_time = 100

    # init angle of arms
    init_angle = np.array((90, 90))
    target_angle = np.array((50, 120))

    # init monitors
    pops_monitor_test = PopMonitor(populations=[PM, S1, StrD1, SNr, VL, M1, Output_Pop_Shoulder, Output_Pop_Elbow],
                                   auto_start=False, sampling_rate=1.)

    # compile model
    folder = f'gain_model_{sim_id}/'
    ann.compile('annarchy/' + folder, clean=exp_args.clean)

    gain_signals = np.arange(0.5, 1.51, 0.25)
    alpha_values = scale_array(gain_signals, 0.2, 1.0)
    check_gain(gain_signals=gain_signals,
               init_thetas=init_angle, sim_time=sim_time,
               target=target_angle, target_angle=True,
               pop_monitor=pops_monitor_test,
               save_path='results/' + folder)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 7))
    plt.subplots_adjust(wspace=0)  # Set horizontal spacing to 0
    for scale, alpha in zip(gain_signals, alpha_values):
        dict = pops_monitor_test.load(folder='results/' + folder + f'gain_scale{scale}/')
        axs[0, 0].plot(dict['r_M1'][-1, :, 0], label=f'D1 scale = {round(scale, 2)}', color='orange', linewidth=2, alpha=alpha)
        axs[0, 1].plot(dict['r_M1'][-1, :, 1], label=f'D1 scale = {round(scale, 2)}', color='blue', linewidth=2, alpha=alpha)

    axs[0, 0].set_xlabel('Encoding neuron $\\theta_{shoulder}$ in [°]', fontsize=12), axs[0, 1].set_xlabel('Encoding neuron $\\theta_{elbow} in [°]$', fontsize=12)
    axs[0, 0].set_ylabel('r', fontsize=12)

    xlabels = [int(ang) if i % 2 == 0 else None for i, ang in enumerate(parameters['motor_orientations'])]
    ylabels = np.round(np.linspace(0, 0.5, 6, endpoint=True), 1)
    axs[0, 0].set_xticks(np.arange(0, parameters['dim_motor'][0]), xlabels)
    axs[0, 1].set_xticks(np.arange(0, parameters['dim_motor'][0]), xlabels),
    axs[0, 0].xaxis.set_label_position('top'), axs[0, 1].xaxis.set_label_position('top')
    axs[0, 0].set_ylim(0, 0.5), axs[0, 1].set_ylim(0, 0.5)
    axs[0, 0].legend(), axs[0, 1].legend()
    axs[0, 0].set_yticks(ylabels, ylabels), axs[0, 1].set_yticks(ylabels, [None for _ in ylabels])

    # norm plots
    ylabels = np.round(np.linspace(0, 1.5, 6, endpoint=True), 1)
    norm_shoulder = []
    norm_elbow = []
    for scale, alpha in zip(gain_signals, alpha_values):
        dict = pops_monitor_test.load(folder='results/' + folder + f'gain_scale{scale}/')
        norm_shoulder.append(np.sqrt(dict['r_Output_Shoulder'][-1, 1]**2 + dict['r_Output_Shoulder'][-1, 2]**2))
        norm_elbow.append(np.sqrt(dict['r_Output_Elbow'][-1, 1]**2 + dict['r_Output_Elbow'][-1, 2]**2))

    axs[1, 0].plot(gain_signals, norm_shoulder, color='orange', linewidth=2)
    axs[1, 1].plot(gain_signals, norm_elbow, color='blue', linewidth=2)
    axs[1, 0].set_xlabel('Gain'), axs[1, 1].set_xlabel('Gain')
    axs[1, 0].set_ylabel('$r_{norm}$')
    axs[1, 0].set_xticks(gain_signals, np.round(gain_signals, 1)), axs[1, 1].set_xticks(gain_signals, np.round(gain_signals, 1))
    axs[1, 0].set_ylim(0, 1.5), axs[1, 1].set_ylim(0, 1.5)
    axs[1, 0].set_yticks(ylabels, ylabels), axs[1, 1].set_yticks(ylabels, [None for _ in ylabels])

    plt.savefig('results/' + folder + 'd1_gain.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    if exp_args.display:
        plt.show()

    plt.close(fig)


