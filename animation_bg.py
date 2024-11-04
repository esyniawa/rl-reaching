import argparse
from typing import Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import ANNarchy as ann
from network.model import *
from experiments import test_reach, test_perturb, training
from kinematics.planar_arms import PlanarArms
from monitoring import PopMonitor


def load_trained_weights(
        save_path: str
):
    ann.load(save_path, populations=False, projections=True)


def plot_2d_populations(
        fig: plt.Figure,
        subplot_id: int,
        y1: np.ndarray,
        y2: np.ndarray,
        horizontal: bool = True,
        ax1_offset_x: float = 0.,
        ax2_offset_x: float = 0.,
        ax1_offset_y: float = 0.,
        ax2_offset_y: float = 0.,
        title: str = '',
        alpha: float = 1.0):
    x = np.arange(y1.shape[0])
    ax1 = fig.add_subplot(3, 3, subplot_id)
    ax2 = fig.add_subplot(3, 3, subplot_id)

    # Create line objects and store them
    line1, = ax1.plot(x, y1, color='blue', linewidth=2, alpha=alpha)
    line2, = ax2.plot(x, y2, color='orange', linewidth=2, alpha=alpha)

    # Set labels
    ax1.set_xlabel('$\\theta_{elbow}$', fontsize=12)
    ax2.set_xlabel('$\\theta_{shoulder}$', fontsize=12)
    ax1.set_ylabel('r', fontsize=12)
    ax2.set_ylabel('r', fontsize=12)

    # Set limits and title
    ax1.set_ylim(0, 1.5)
    ax2.set_ylim(0, 1.5)
    ax2.set_title(title, fontsize=18)

    if not horizontal:
        ax2.xaxis.set_label_position('top')

    # Set positions
    pos = ax1.get_position()
    if horizontal:
        ax1.set_position([pos.x0 + ax1_offset_x, pos.y0 + ax1_offset_y, 0.08, 0.08])
        ax2.set_position([pos.x0 + ax2_offset_x + 0.1, pos.y0 + ax2_offset_y, 0.08, 0.08])
    else:
        ax1.set_position([pos.x0 + ax1_offset_x, pos.y0 + ax1_offset_y, 0.08, 0.08])
        ax2.set_position([pos.x0 + ax2_offset_x, pos.y0 + ax2_offset_y + 0.1, 0.08, 0.08])

    # Set general formatting
    for ax in [ax1, ax2]:
        ax.set_facecolor('none')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)

    return ax1, ax2


def update(frame, monitor_dict, s1_plot, pm_plot, d1_plots, snr_axs, vl_axs, m1_axs, cm_axs, snc_plot,
           readout_shoulder_plot, readout_elbow_plot):
    """Update function for the animation"""
    x = np.arange(monitor_dict['r_SNr'].shape[1])  # Get x values for line plots

    # Update S1
    s1_plot.set_array(monitor_dict['r_S1'][frame].astype(np.float64))

    # Update PM
    pm_plot.set_array(monitor_dict['r_PM'][frame].astype(np.float64))

    # Update D1 subplots
    for plot_idx, d1_plot in enumerate(d1_plots):
        d1_plot.set_array(monitor_dict['r_StrD1'][frame, :, :, plot_idx].astype(np.float64))

    # Update populations using set_data
    population_updates = [
        (snr_axs, monitor_dict['r_SNr'][frame]),
        (vl_axs, monitor_dict['r_VL'][frame]),
        (m1_axs, monitor_dict['r_M1'][frame]),
        (cm_axs, monitor_dict['r_CM'][frame])
    ]

    for axs, data in population_updates:
        # Update each line's data
        axs[0].lines[0].set_data(x, data[:, 0])
        axs[1].lines[0].set_data(x, data[:, 1])

    # Update SNc
    snc_plot[0].set_xdata(frame)
    snc_plot[0].set_ydata(monitor_dict['r_SNc'][frame])

    # Update readout plots
    # Shoulder readout
    rad_shoulder = (0, np.radians(monitor_dict['r_Output_Shoulder'][frame, 0]))
    r_shoulder = (0, np.sqrt(monitor_dict['r_Output_Shoulder'][frame, 1] ** 2 +
                             monitor_dict['r_Output_Shoulder'][frame, 2] ** 2))
    readout_shoulder_plot[0].set_data(rad_shoulder, r_shoulder)

    # Elbow readout
    rad_elbow = (0, np.radians(monitor_dict['r_Output_Elbow'][frame, 0]))
    r_elbow = (0, np.sqrt(monitor_dict['r_Output_Elbow'][frame, 1] ** 2 +
                          monitor_dict['r_Output_Elbow'][frame, 2] ** 2))
    readout_elbow_plot[0].set_data(rad_elbow, r_elbow)

    # Return all artists that were modified
    artists = [s1_plot, pm_plot, *d1_plots]
    for axs, _ in population_updates:
        artists.extend([axs[0].lines[0], axs[1].lines[0]])
    artists.extend(snc_plot)
    artists.extend([readout_shoulder_plot[0], readout_elbow_plot[0]])

    return tuple(artists)


def create_animation(fig, monitor_dict, save_path=None):
    """
    Create and save the animation

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure containing all subplots
    monitor_dict : dict
        Dictionary containing population activities over time
    save_path : str, optional
        Path to save the animation
    """
    # Get S1 and PM plots
    s1_plot = [obj for obj in fig.axes[0].get_children()
               if isinstance(obj, plt.matplotlib.image.AxesImage)][0]
    pm_plot = [obj for obj in fig.axes[1].get_children()
               if isinstance(obj, plt.matplotlib.image.AxesImage)][0]

    # Get D1 plots - they start after S1 and PM (first 2 axes)
    d1_start_idx = 2
    d1_end_idx = d1_start_idx + 32  # 8x4 = 32 subplots
    d1_plots = []
    for ax in fig.axes[d1_start_idx:d1_end_idx]:
        d1_plot = [obj for obj in ax.get_children()
                   if isinstance(obj, plt.matplotlib.image.AxesImage)]
        if d1_plot:
            d1_plots.append(d1_plot[0])

    # Get population axes
    snr_axs = fig.axes[d1_end_idx:d1_end_idx + 2]
    vl_axs = fig.axes[d1_end_idx + 2:d1_end_idx + 4]
    m1_axs = fig.axes[d1_end_idx + 4:d1_end_idx + 6]

    # Get readout plots
    readout_shoulder_plot = [line for line in fig.axes[d1_end_idx + 6].get_lines()]
    readout_elbow_plot = [line for line in fig.axes[d1_end_idx + 7].get_lines()]

    cm_axs = fig.axes[d1_end_idx + 8:d1_end_idx + 10]

    # Get SNc plot
    snc_plot = [line for line in fig.axes[-1].get_lines()
                if line.get_marker() == 'x']

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(monitor_dict['r_S1']),
        fargs=(monitor_dict, s1_plot, pm_plot, d1_plots,
               snr_axs, vl_axs, m1_axs, cm_axs, snc_plot,
               readout_shoulder_plot, readout_elbow_plot),
        interval=1,  # 10ms between frames
        blit=True
    )

    writer = animation.FFMpegWriter(fps=30)
    # Save animation if path is provided
    if save_path:
        anim.save(save_path, writer=writer)

    return anim


def create_d1_subplots(fig, subplot_position, inner_rows=8, inner_cols=4):
        """
        Create D1 subplots using GridSpec

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The main figure
        subplot_position : int
            Position in the main 3x3 grid where D1 should be placed
        inner_rows : int
            Number of rows in D1 grid
        inner_cols : int
            Number of columns in D1 grid

        Returns:
        --------
        list of matplotlib.axes.Axes
            List of axes for D1 subplots
        """
        # Create a gridspec for the main 3x3 layout
        outer_grid = gridspec.GridSpec(3, 3)

        # Get the position for D1 in the main grid
        row = (subplot_position - 1) // 3
        col = (subplot_position - 1) % 3

        # Create a subgridspec for D1
        inner_grid = gridspec.GridSpecFromSubplotSpec(inner_rows, inner_cols,
                                                      subplot_spec=outer_grid[row, col],
                                                      wspace=0.0, hspace=0.0)

        # Create the subplots
        d1_axes = []
        for i in range(inner_rows * inner_cols):
            ax = fig.add_subplot(inner_grid[i // inner_cols, i % inner_cols])
            ax.set_xticks([])
            ax.set_yticks([])
            d1_axes.append(ax)

        return d1_axes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1, help='Simulation ID')
    parser.add_argument('--pretrained_data_path', type=str, default=None,
                        help='Path to synaptic weights of pre-trained model')
    parser.add_argument('--condition', type=str, default='test', choices=['training', 'test', 'test_pert'],
                        help='Training or testing condition')
    parser.add_argument('--clean', type=bool, default=False, help='Clean ANNarchy compilation of the model')
    parser.add_argument('--n_trials', type=int, default=4)

    args = parser.parse_args()

    # simulations informations
    sim_id = args.id

    # init angle of arms
    init_angle = np.array((90., 90.))
    my_arms = PlanarArms(init_angles_left=init_angle, init_angles_right=init_angle, radians=False)
    pops_monitor = PopMonitor(populations=[PM, S1, StrD1, SNr, VL, M1, SNc, CM, Output_Pop_Shoulder, Output_Pop_Elbow],
                              auto_start=False, sampling_rate=1.)

    # compile model
    folder = f'animation_{sim_id}/'
    ann.compile('annarchy/' + folder, clean=args.clean)

    # load pre_trained cons
    if args.pretrained_data_path is not None:
        data_path = args.pretrained_data_path
        if data_path[-1] != '/':
            data_path += '/'

        print('Loading pretrained weights...')
        save_path_weights = f'results/training_run_model_{sim_id}/{data_path}synaptic_weights.npz'
        load_trained_weights(save_path_weights)
        folder = f'animation_{sim_id}_pretrained/'

    # run
    print(f'Starting simulation with condition: {args.condition} ...')
    if args.condition == 'training':
        if os.path.isfile(f'results/{args.condition}_{folder}data.npz'):
            results = np.load(f'results/{args.condition}_{folder}data.npz')
        else:
            training(N_trials=args.n_trials,
                     init_angle=init_angle,
                     save_path=folder,
                     pop_monitor=pops_monitor,
                     con_monitor=None, )
            results = np.load(f'results/{args.condition}_{folder}data.npz')
    elif args.condition == 'test':
        if os.path.isfile(f'results/{args.condition}_{folder}data.npz'):
            results = np.load(f'results/{args.condition}_{folder}data.npz')
        else:
            test_reach(init_angle=init_angle,
                       save_path=folder,
                       pop_monitor=pops_monitor,
                       animate_populations=False,
                       test_condition='cube', )
            results = np.load(f'results/{args.condition}_{folder}data.npz')
    elif args.condition == 'test_pert':
        if os.path.isfile(f'results/{args.condition}_{folder}data.npz'):
            results = np.load(f'results/{args.condition}_{folder}data.npz')
        else:
            test_perturb(init_angle=init_angle,
                         N_trials=args.n_trials,
                         save_path=folder,
                         pop_monitor=pops_monitor,
                         animate_populations=False, )
            results = np.load(f'results/{args.condition}_{folder}data.npz')
    else:
        raise NotImplementedError('Invalid condition')

    # load monitors
    monitor_dict = pops_monitor.load(f'results/{args.condition}_{folder}/')

    fig = plt.figure(figsize=(18, 14))

    hspace = 0.05
    vspace = 0.05

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=hspace, wspace=vspace)
    max_time = monitor_dict['r_M1'].shape[0]

    # S1
    s1_ax = fig.add_subplot(3, 3, 2)
    s1_plot = s1_ax.imshow(monitor_dict['r_S1'][0].astype(np.float64), vmin=0, vmax=1, cmap='Blues',
                           origin='lower', interpolation='none')
    s1_ax.set_title('S1', fontsize=18)
    s1_ax.set_xticks([]), s1_ax.set_yticks([])
    s1_ax.set_ylabel('$\\theta_{elbow}$'), s1_ax.set_xlabel('$\\theta_{shoulder}$')
    pos = s1_ax.get_position()
    s1_ax.set_position([pos.x0 + 0.03, pos.y0 + 0.1, 0.15, 0.15])

    # PM
    pm_ax = fig.add_subplot(3, 3, 4)
    pm_plot = pm_ax.imshow(monitor_dict['r_PM'][0].astype(np.float64), vmin=0, vmax=1, cmap='Blues',
                           origin='lower', interpolation='none')
    pm_ax.set_xticks([])
    pm_ax.set_yticks([])
    pm_ax.set_ylabel('$y$')
    pm_ax.set_xlabel('$x$')
    pm_ax.set_title('PM', fontsize=18)
    pos = pm_ax.get_position()
    pm_ax.set_position([pos.x0+0.025, pos.y0 + 0.05, 0.15, 0.15])

    # D1
    # Replace your D1 subplot creation with:
    d1_axes = create_d1_subplots(fig, subplot_position=5)  # 5 is the center position in 3x3 grid

    # Create the D1 plots
    d1_plots = []
    for ax, result in zip(d1_axes, np.rollaxis(monitor_dict['r_StrD1'][0], -1)):
        p = ax.imshow(result.astype(np.float64), vmin=0, vmax=1, cmap='Blues',
                      origin='lower', interpolation='none')
        d1_plots.append(p)

    # SNr
    snr_axs = plot_2d_populations(subplot_id=8, y1=monitor_dict['r_SNr'][0, :, 0], y2=monitor_dict['r_SNr'][0, :, 1],
                                  fig=fig, horizontal=False, title='SNr',
                                  ax1_offset_x=0.1, ax2_offset_x=0.1, ax1_offset_y=-0.05, ax2_offset_y=-0.05,
                                  alpha=0.5)

    # VL
    vl_axs = plot_2d_populations(subplot_id=9, y1=monitor_dict['r_VL'][0, :, 0], y2=monitor_dict['r_VL'][0, :, 1],
                                 fig=fig, horizontal=False, title='VL',
                                 ax1_offset_x=0.1, ax2_offset_x=0.1, ax1_offset_y=-0.05, ax2_offset_y=-0.05,
                                 alpha=0.5)

    # M1
    m1_axs = plot_2d_populations(subplot_id=6, y1=monitor_dict['r_M1'][0, :, 0], y2=monitor_dict['r_M1'][0, :, 1],
                                 fig=fig, horizontal=False, title='M1',
                                 ax1_offset_x=0.1, ax2_offset_x=0.1, ax1_offset_y=0.03, ax2_offset_y=0.03, alpha=0.5)

    readout_shoulder = fig.add_subplot(3, 3, 6, projection='polar')
    pos = readout_shoulder.get_position()
    readout_shoulder.set_position([pos.x0 + 0.18, pos.y0 + 0.15, 0.08, 0.08])
    rad = (0, np.radians(monitor_dict['r_Output_Shoulder'][0, 0]))
    r = (0, np.sqrt(monitor_dict['r_Output_Shoulder'][0, 1] ** 2 + monitor_dict['r_Output_Shoulder'][0, 2] ** 2))
    readout_shoulder_plot = readout_shoulder.plot(rad, r, c='orange')
    readout_shoulder.set_ylim([0, 2])

    readout_elbow = fig.add_subplot(3, 3, 6, projection='polar')
    pos = readout_elbow.get_position()
    readout_elbow.set_position([pos.x0 + 0.18, pos.y0, 0.08, 0.08])
    rad = (0, np.radians(monitor_dict['r_Output_Elbow'][0, 0]))
    r = (0, np.sqrt(monitor_dict['r_Output_Elbow'][0, 1] ** 2 + monitor_dict['r_Output_Elbow'][0, 2] ** 2))
    readout_elbow_plot = readout_elbow.plot(rad, r, c='blue')
    readout_elbow.set_ylim([0, 2])

    # CM
    cm_axs = plot_2d_populations(subplot_id=3, y1=monitor_dict['r_CM'][0, :, 0], y2=monitor_dict['r_CM'][0, :, 1],
                                 fig=fig, horizontal=False, title='CM',
                                 ax1_offset_x=0.1, ax2_offset_x=0.1, ax1_offset_y=0.1, ax2_offset_y=0.1, alpha=0.5)

    # SNc
    snc_ax = fig.add_subplot(3, 3, 7)
    res_max = np.amax(monitor_dict['r_SNc'])

    # plotting
    snc_ax.plot(monitor_dict['r_SNc'])
    snc_plot = snc_ax.plot(monitor_dict['r_SNc'][0], marker='x', color='r')
    snc_ax.set_ylabel('$r^{DA}$')
    snc_ax.set_title('SNc', fontsize=18)
    snc_ax.set_xlabel('t', loc='right')
    snc_ax.set_ylim(0, res_max + 0.1)
    pos = snc_ax.get_position()
    snc_ax.set_position([pos.x0 + 0.05, pos.y0 + 0.01, 0.1, 0.1])

    # Create and show animation
    anim = create_animation(fig, monitor_dict, save_path='analysis/training.mp4')
