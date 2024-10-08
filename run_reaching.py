from network.model import *
from kinematics.planar_arms import PlanarArms
from experiments import training, test_reach, test_perturb
from monitoring import PopMonitor, ConMonitor
import argparse

if __name__ == '__main__':
    exp_parser = argparse.ArgumentParser()
    exp_parser.add_argument('--id', type=int, default=0, help='Simulation ID')
    exp_parser.add_argument('--test_pert', type=bool, default=False,
                            help='Whether to test perturbation or not')
    exp_parser.add_argument('--test_reach_condition', type=str,
                            choices=['cube', 'random', 'circle'],
                            default='random',
                            help='Testing condition for reaching')
    exp_parser.add_argument('--con_monitor', type=bool, default=False)
    exp_parser.add_argument('--clean', type=bool, default=False, help='Clean ANNarchy compilation of the model')
    exp_parser.add_argument('--animate_arms', type=bool, default=False)
    exp_args = exp_parser.parse_args()

    # number of training trials
    training_trials = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 52_000)
    test_condition = exp_args.test_reach_condition

    # init angle of arms
    init_angle = np.array((90, 90))
    if exp_args.animate_arms:
        my_arms = PlanarArms(init_angles_left=init_angle, init_angles_right=init_angle, radians=False)
    else:
        my_arms = None

    # init monitors
    PopMonitor_training = PopMonitor(populations=[SNr, VL, M1, Output_Pop_Shoulder, Output_Pop_Elbow],
                                     auto_start=False, sampling_rate=100.)
    PopMonitor_testing = PopMonitor(populations=[PM, S1, StrD1, SNr, VL, M1, Output_Pop_Shoulder, Output_Pop_Elbow],
                                    auto_start=False, sampling_rate=50.)

    if exp_args.con_monitor:
        ConMonitor_training = ConMonitor(connections=[PM_StrD1],
                                         reshape_pre=[False],
                                         reshape_post=[True])
    else:
        ConMonitor_training = None

    # save data in this folder
    sim_id = exp_args.id
    folder = f'run_model_{sim_id}/'

    # compile model
    ann.compile('annarchy/' + folder, clean=exp_args.clean)

    for N_training_trials in training_trials:
        print(f'Sim {sim_id}: Training for {N_training_trials}...')

        subfolder = f'model_{N_training_trials}/'
        training(N_trials=N_training_trials,
                 init_angle=init_angle,
                 save_path=folder + subfolder,
                 pop_monitor=PopMonitor_training,
                 con_monitor=ConMonitor_training,
                 animate_populations=False,
                 plot_error=True,
                 save_synapses=True)

        print(f'Sim {sim_id}: Testing ...')
        test_reach(init_angle=init_angle,
                   save_path=folder + subfolder,
                   pop_monitor=PopMonitor_testing,
                   test_condition=test_condition,
                   plot_error=True,
                   animate_populations=True,
                   arms_model=my_arms,
                   num_random_points=100)

        if exp_args.animate_arms:
            my_arms.reset_all()

        if exp_args.test_pert:
            test_perturb(init_angle=init_angle,
                         N_trials=20,
                         save_path=folder + subfolder,
                         pop_monitor=PopMonitor_testing,
                         plot_error=True,
                         animate_populations=True,
                         arms_model=my_arms)

            # reset arms and monitors
            if exp_args.animate_arms:
                my_arms.reset_all()

    print('Done!')
