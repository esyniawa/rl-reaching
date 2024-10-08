import numpy as np
from os import path, makedirs
from network.model import *
from create_inputs import (train_position,
                           train_fixed_position,
                           test_movement,
                           test_perturbation,
                           generate_random_coordinate)
from utils import safe_save
from kinematics.planar_arms import PlanarArms

from monitoring import PopMonitor, ConMonitor
import matplotlib.pyplot as plt


def training(N_trials: int,
             init_angle: np.ndarray,
             save_path: str,
             reward_time: int = 450,
             wait_time: int = 50,
             pop_monitor: PopMonitor | None = None,
             con_monitor: ConMonitor | None = None,
             save_synapses: bool = False,
             animate_populations: bool = False,
             plot_error: bool = False) -> None:
    # look for possible errors
    if pop_monitor is None and animate_populations is True:
        raise ValueError('PopMonitor must be specified if animate populations is True')
    if save_path[-1] != '/':
        save_path += '/'

    # initialize monitors
    if pop_monitor is not None:
        pop_monitor.start()

    if con_monitor is not None:
        con_monitor.extract_weights()

    training_infos = {
        'init_angle': init_angle,
        'target_angle': [],
        'output_angle': [],
        'sim_time': [],
        'error': [],
    }

    for trial in range(N_trials):
        init_angle, out, sim_time = train_position(current_thetas=init_angle,
                                                   t_reward=reward_time,
                                                   t_wait=wait_time)

        training_infos['target_angle'].append(init_angle)
        training_infos['output_angle'].append(out)
        training_infos['sim_time'].append(sim_time)
        training_infos['error'].append(
            np.linalg.norm(
                PlanarArms.forward_kinematics(arm=parameters['moving_arm'], thetas=init_angle, radians=False)[:, -1]
                - PlanarArms.forward_kinematics(arm=parameters['moving_arm'], thetas=out, radians=False)[:, -1]))

    # save infos
    if not path.exists('results/' + 'training_' + save_path):
        makedirs('results/' + 'training_' + save_path)
    np.savez('results/' + 'training_' + save_path + 'data.npz', **training_infos, allow_pickle=True)

    # saving data
    # rates
    if pop_monitor is not None:
        pop_monitor.save(folder='results/' + 'training_' + save_path, delete=True)
        pop_monitor.stop()

    # weights
    if con_monitor is not None:
        con_monitor.extract_weights()
        con_monitor.save_cons(folder='results/' + 'training_' + save_path)
        con_monitor.reset()

    if save_synapses:
        ann.save('results/' + 'training_' + save_path + 'synaptic_weights.npz', populations=False)

    if plot_error:
        fig, ax = plt.subplots()
        ax.plot(np.array(training_infos['error']))
        plt.savefig('results/' + 'training_' + save_path + 'error.png')
        plt.close(fig)

    if animate_populations:
        pop_names = [name for name in pop_monitor.get_population_names]
        pop_plot_types = [parameters['pop_plot_types'][name] for name in pop_names]

        PopMonitor.load_and_animate(pops=pop_names,
                                    plot_types=pop_plot_types,
                                    folder='results/' + 'training_' + save_path)


def training_fix_points(points: list[np.ndarray],
                        init_angle: np.ndarray,
                        save_path: str,
                        reward_time: int = 450,
                        wait_time: int = 50,
                        pop_monitor: PopMonitor | None = None,
                        con_monitor: ConMonitor | None = None,
                        save_synapses: bool = False,
                        animate_populations: bool = False) -> None:

    # look for possible errors
    if pop_monitor is None and animate_populations is True:
        raise ValueError('PopMonitor must be specified if animate populations is True')
    if save_path[-1] != '/':
        save_path += '/'
    assert init_angle.size == 2, 'init_angle must be a 2D array'

    # initialize monitors
    if pop_monitor is not None:
        pop_monitor.start()

    if con_monitor is not None:
        con_monitor.extract_weights()

    for point in points:
        init_angle = train_fixed_position(current_thetas=init_angle,
                                          goal=point,
                                          t_reward=reward_time,
                                          t_wait=wait_time)

    # saving data
    # rates
    if pop_monitor is not None:
        pop_monitor.save(folder='results/' + 'training_' + save_path, delete=True)
        pop_monitor.stop()

    # weights
    if con_monitor is not None:
        con_monitor.extract_weights()
        con_monitor.save_cons(folder='results/' + 'training_' + save_path)
        con_monitor.reset()

    if save_synapses:
        ann.save('results/' + 'training_' + save_path + 'synaptic_weights.npz', populations=False)

    if animate_populations:
        pop_names = [name for name in pop_monitor.get_population_names]
        pop_plot_types = [parameters['pop_plot_types'][name] for name in pop_names]

        PopMonitor.load_and_animate(pops=pop_names,
                                    plot_types=pop_plot_types,
                                    folder='results/' + 'training_' + save_path)


def test_reach(init_angle: np.ndarray,
               save_path: str,
               movement_time: int = 450,
               wait_time: int = 50,
               pop_monitor: PopMonitor | None = None,
               plot_error: bool = True,
               test_condition: str = 'cube',
               animate_populations: bool = True,
               arms_model: PlanarArms | None = None,
               show_plot: bool = False,
               scale_pm: float = 1.0,
               scale_s1: float = 1.0,
               num_random_points: int = 100) -> None:
    # look for possible errors
    if pop_monitor is None and animate_populations is True:
        raise ValueError('Monitor populations must be specified if animate populations is True')
    if save_path[-1] != '/':
        save_path += '/'

    # get points to follow
    if test_condition == 'cube':
        points_to_follow = parameters['test_trajectory_cube']
    elif test_condition == 'circle':
        points_to_follow = parameters['test_trajectory_circle']
    elif test_condition == 'random':
        points_to_follow = []
        for _ in range(num_random_points):
            _, points = generate_random_coordinate()
            points_to_follow.append(points)
    else:
        raise ValueError('Unknown test condition')

    # start monitors
    if pop_monitor is not None:
        pop_monitor.start()

    test_infos = {
        'init_angle': [],
        'target_angle': [],
        'target_pos': [],
        'output_angle': [],
        'sim_time': [],
        'error': [],
    }

    for point in points_to_follow:
        test_infos['init_angle'].append(init_angle)
        init_angle, out, sim_time = test_movement(current_thetas=init_angle,
                                                  point_to_reach=point,
                                                  scale_pm=scale_pm,
                                                  scale_s1=scale_s1,
                                                  t_movement=movement_time,
                                                  t_wait=wait_time,
                                                  arms_model=arms_model)

        test_infos['target_angle'].append(init_angle)
        test_infos['target_pos'].append(point)
        test_infos['output_angle'].append(out)
        test_infos['sim_time'].append(sim_time)
        test_infos['error'].append(
            np.linalg.norm(
                point - PlanarArms.forward_kinematics(arm=parameters['moving_arm'], thetas=out, radians=False)[:, -1]))

    # saving data
    if not path.exists('results/' + 'test_' + save_path):
        makedirs('results/' + 'test_' + save_path)
    np.savez('results/' + 'test_' + save_path + 'data.npz', **test_infos)

    if plot_error:
        fig, ax = plt.subplots()
        ax.plot(np.array(test_infos['error']))
        plt.savefig('results/' + 'test_' + save_path + 'error.png')
        if show_plot:
            plt.show()

        plt.close(fig)

    # rates
    if pop_monitor is not None:
        pop_monitor.save(folder='results/' + 'test_' + save_path, delete=True)
        pop_monitor.stop()

    # TODO: Weird behavior if the format is gif, not mp4
    if arms_model is not None:
        arms_model.plot_trajectory(save_name=f'results/test_{save_path}trajectory_reaching.mp4',
                                   points=points_to_follow)
        arms_model.reset_all()

    if animate_populations:
        pop_names = [name for name in pop_monitor.get_population_names]
        pop_plot_types = [parameters['pop_plot_types'][name] for name in pop_names]

        if show_plot:
            PopMonitor.load_and_animate(pops=pop_names,
                                        plot_types=pop_plot_types,
                                        folder='results/' + 'test_' + save_path)
        else:
            PopMonitor.load_and_animate(pops=pop_names,
                                        plot_types=pop_plot_types,
                                        folder='results/' + 'test_' + save_path,
                                        save_name=f'results/test_{save_path}pops_reaching.gif')


def test_perturb(init_angle: np.ndarray,
                 N_trials: int,
                 save_path: str,
                 pre_movement_time: int = 50,
                 movement_time: int = 450,
                 wait_time: int = 50,
                 pop_monitor: PopMonitor | None = None,
                 perturbation_scale: float = 10.0,  # in [°]
                 plot_error: bool = True,
                 arms_model: PlanarArms | None = None,
                 animate_populations: bool = False,
                 show_plots: bool = False,
                 scale_pm: float = 1.0,
                 scale_s1: float = 1.0) -> None:

    # look for possible errors
    if pop_monitor is None and animate_populations is True:
        raise ValueError('Monitor populations must be specified if animate populations is True')
    if save_path[-1] != '/':
        save_path += '/'

    # start monitors
    if pop_monitor is not None:
        pop_monitor.start()

    # test
    test_infos = {
        'init_angle': [],
        'target_angle': [],
        'target_pos': [],
        'output_before_pert': [],
        'output_after_pert': [],
        'sim_time_before_pert': [],
        'sim_time_after_pert': [],
        'perturbation_shoulder': [],
        'perturbation_elbow': [],
        'error_before_pert': [],
        'error_after_pert': [],
    }
    points = []

    for trial in range(N_trials):
        _, random_point = generate_random_coordinate()
        points.append(random_point)

        # calc perturbation
        pert_sh, pert_el = np.random.uniform(-perturbation_scale, perturbation_scale, size=2)

        # run perturbation trial
        test_infos['init_angle'].append(init_angle)
        init_angle, out_1, out_2, sim_time_1, sim_time_2, pert_sh, pert_el = test_perturbation(
            current_thetas=init_angle,
            point_to_reach=random_point,
            perturbation_shoulder=pert_sh,
            perturbation_elbow=pert_el,
            scale_pm=scale_pm,
            scale_s1=scale_s1,
            t_init=pre_movement_time,
            t_movement=movement_time,
            t_wait=wait_time,
            arms_model=arms_model)

        test_infos['target_angle'].append(init_angle)
        test_infos['target_pos'].append(random_point)
        test_infos['output_before_pert'].append(out_1)
        test_infos['output_after_pert'].append(out_2)
        test_infos['sim_time_before_pert'].append(sim_time_1)
        test_infos['sim_time_after_pert'].append(sim_time_2)
        test_infos['perturbation_shoulder'].append(pert_sh)
        test_infos['perturbation_elbow'].append(pert_el)
        test_infos['error_before_pert'].append(
            np.linalg.norm(random_point - PlanarArms.forward_kinematics(arm=parameters['moving_arm'], thetas=out_1, radians=False)[:, -1]))
        test_infos['error_after_pert'].append(
            np.linalg.norm(random_point - PlanarArms.forward_kinematics(arm=parameters['moving_arm'], thetas=out_2, radians=False)[:, -1]))

    # saving data
    if not path.exists('results/' + 'test_pert_' + save_path):
        makedirs('results/' + 'test_pert_' + save_path)
    np.savez('results/' + 'test_pert_' + save_path + 'data.npz', **test_infos)

    if plot_error:
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(np.array(test_infos['error_before_pert']), label='Before perturbation')
        axs[1].plot(np.array(test_infos['error_after_pert']), label='After perturbation')
        plt.legend()
        plt.savefig('results/' + 'test_pert_' + save_path + 'error.png')
        if show_plots:
            plt.show()
        plt.close(fig)

    # rates
    if pop_monitor is not None:
        pop_monitor.save(folder='results/' + 'test_pert_' + save_path, delete=True)
        pop_monitor.stop()

    # TODO: Weird behavior if the format is gif, not mp4
    if arms_model is not None:
        arms_model.plot_trajectory(points=points,
                                   save_name=f'results/test_pert_{save_path}trajectory_perturbation.mp4',
                                   frames_per_sec=20)
        arms_model.reset_all()

    # animate population activity
    if animate_populations:
        pop_names = [name for name in pop_monitor.get_population_names]
        pop_plot_types = [parameters['pop_plot_types'][name] for name in pop_names]

        if show_plots:
            PopMonitor.load_and_animate(pops=pop_names,
                                        plot_types=pop_plot_types,
                                        folder='results/' + 'test_pert_' + save_path)
        else:
            PopMonitor.load_and_animate(pops=pop_names,
                                        plot_types=pop_plot_types,
                                        folder='results/' + 'test_pert_' + save_path,
                                        save_name=f'results/test_pert_{save_path}pops_perturbation.gif')


def check_gain(
        gain_signals: tuple,
        init_thetas: np.ndarray,
        target: np.ndarray,
        save_path: str,
        pop_monitor: PopMonitor | None = None,
        target_angle: bool = False,
        sim_time: float = 150.
):
    if PopMonitor is not None:
        pop_monitor.start()

    if target_angle:
        target = PlanarArms.forward_kinematics(arm=parameters['moving_arm'], thetas=target, radians=False)[:, -1]

    print('Train position...')
    training_fix_points(points=[target], init_angle=init_thetas, save_path=save_path)

    print('Test movement...')
    for scale in gain_signals:
        test_movement(current_thetas=init_thetas, point_to_reach=target, scale_s1=scale, t_movement=sim_time)

        if PopMonitor is not None:
            pop_monitor.save(folder=save_path + f'gain_scale{scale}/', delete=True)
