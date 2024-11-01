import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from network.model import *
from experiments import test_reach, test_perturb
from kinematics.planar_arms import PlanarArms
from monitoring import PopMonitor


def load_trained_weights(
        save_path: str
):
    ann.load(save_path, populations=False, projections=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1, help='Simulation ID')
    parser.add_argument('--data_path', type=str, default='model_32000',
                        help='Path to synaptic weights of pre-trained model')
    parser.add_argument('--test_pert', type=bool, default=False,
                        help='Whether to test perturbation or not')
    parser.add_argument('--test_reach_condition', type=str,
                        choices=['cube', 'random', 'circle'],
                        default='cube',
                        help='Testing condition for reaching')
    parser.add_argument('--display', type=bool, default=True)
    parser.add_argument('--clean', type=bool, default=False, help='Clean ANNarchy compilation of the model')
    args = parser.parse_args()

    # simulations informations
    sim_id = args.id
    data_path = args.data_path
    if data_path[-1] != '/':
        data_path += '/'

    save_path_weights = f'results/training_run_model_{sim_id}/{args.data_path}synaptic_weights.npz'

    # init angle of arms
    init_angle = np.array((90., 90.))
    my_arms = PlanarArms(init_angles_left=init_angle, init_angles_right=init_angle, radians=False)
    pops_monitor = PopMonitor(populations=[PM, S1, StrD1, SNr, VL, M1, SNc, Output_Pop_Shoulder, Output_Pop_Elbow],
                              auto_start=False, sampling_rate=5.)

    # folder to save results
    folder = f'pretrained_{sim_id}/'

    # compile model
    ann.compile('annarchy/' + folder, clean=args.clean)

    # load pre_trained cons
    print('Loading weights...')
    load_trained_weights(save_path_weights)

    print('Testing reaching...')
    test_reach(init_angle=init_angle,
               save_path=folder,
               pop_monitor=pops_monitor,
               plot_error=True,
               animate_populations=True,
               show_plot=args.display,
               test_condition=args.test_reach_condition,
               arms_model=my_arms)

    if args.test_pert:
        test_perturb(init_angle=init_angle,
                     N_trials=10,
                     save_path=folder,
                     pop_monitor=pops_monitor,
                     plot_error=True,
                     animate_populations=True,
                     arms_model=my_arms,
                     show_plots=args.display)

    print('Done!')