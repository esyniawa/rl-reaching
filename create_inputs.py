import matplotlib.pyplot as plt
import numpy as np

from network.params import parameters
from network.utils import bivariate_gauss, circ_gauss, gauss
from network.model import *

from kinematics.planar_arms import PlanarArms
from utils import generate_random_coordinate


def make_inputs(current_angles: np.ndarray,
                end_point: np.ndarray,
                next_angles: np.ndarray | None = None,
                degrees: bool = True):

    if not degrees:
        current_angles = np.degrees(current_angles)
        if next_angles is not None:
            next_angles = np.degrees(next_angles)

    # calculate new thetas to the end point. Returns angle in radians
    if next_angles is not None:
        new_thetas = next_angles
    else:
        new_thetas = PlanarArms.inverse_kinematics(
            arm=parameters['moving_arm'],
            end_effector=end_point,
            starting_angles=current_angles,
            radians=False
        )
        new_thetas = np.degrees(new_thetas)

    # calculate pm input
    input_pm = bivariate_gauss(xy=parameters['state_pm'],
                               mu=end_point,
                               sigma=parameters['sig_pm'])

    # calculate s1 input
    input_s1 = bivariate_gauss(xy=parameters['state_s1'],
                               mu=current_angles,
                               sigma=parameters['sig_s1'])

    # calculate motor input
    input_cm = np.zeros(parameters['dim_motor'])
    for layer in range(parameters['dim_motor'][1]):
        input_cm[:, layer] = gauss(mu=new_thetas[layer],
                                   sigma=parameters['sig_m1'],
                                   x=parameters['motor_orientations'])

    return input_pm, input_s1, input_cm, new_thetas


def trial(input_pm: np.ndarray,
          input_s1: np.ndarray,
          input_cm: np.ndarray | None,
          t_wait: float,
          t_sim: float,
          training: bool = True,
          reset: bool = True):

    if not training:
        ann.disable_learning()
    else:
        ann.enable_learning()

    # simulation state
    if t_wait > 0.:
        SNc.firing = 0
        PM.baseline = 0
        S1.baseline = 0
        CM.baseline = 0
        ann.simulate(t_wait)

    # send reward and set inputs
    if training:
        SNc.firing = 1
    PM.baseline = input_pm
    S1.baseline = input_s1
    if input_cm is not None:
        CM.baseline = input_cm
    sim_time = ann.simulate_until(t_sim, population=SNr)

    out = np.array((Output_Pop_Shoulder.r[0], Output_Pop_Elbow.r[0]))

    # reset
    if reset:
        ann.reset(populations=True, monitors=False)

    return sim_time, out


def train_position(current_thetas: np.ndarray,
                   t_reward: float = 400.,
                   t_wait: float = 50.):

    new_thetas, new_position = generate_random_coordinate()

    # make input
    base_pm, base_s1, base_m1, new_thetas = make_inputs(current_angles=current_thetas,
                                                        next_angles=new_thetas,
                                                        end_point=new_position)

    sim_time, out = trial(input_pm=base_pm,
                          input_s1=base_s1,
                          input_cm=base_m1,
                          t_wait=t_wait,
                          t_sim=t_reward,
                          training=True)

    return new_thetas, out, sim_time


def train_fixed_position(current_thetas: np.ndarray,
                         goal: np.ndarray,
                         t_reward: float = 400.,
                         t_wait: float = 50.):

    base_pm, base_s1, base_m1, new_thetas = make_inputs(current_angles=current_thetas,
                                                        end_point=goal)

    trial(input_pm=base_pm,
          input_s1=base_s1,
          input_cm=base_m1,
          t_wait=t_wait,
          t_sim=t_reward,
          training=True)

    return new_thetas


def test_movement(current_thetas: np.ndarray,
                  point_to_reach: np.ndarray,
                  scale_pm: float = 1.0,
                  scale_s1: float = 1.0,
                  t_movement: float = 400.,
                  t_wait: float = 50.,
                  arms_model: PlanarArms | None = None):

    # make inputs for PM
    input_pm, input_s1, _, new_thetas = make_inputs(current_angles=current_thetas,
                                                    end_point=point_to_reach)

    _, output_theta = trial(input_pm=input_pm * scale_pm,
                            input_s1=input_s1 * scale_s1,
                            input_cm=None,
                            t_wait=t_wait,
                            t_sim=t_movement,
                            training=False)

    # update current angles in arm model
    error = np.array((output_theta[0] - new_thetas[0], output_theta[1] - new_thetas[1]))

    output_theta = PlanarArms.clip_values(output_theta, radians=False)
    if arms_model is not None:
        arms_model.change_angle(arm='right', new_thetas=output_theta, radians=False)

    return new_thetas, error


def test_perturbation(current_thetas: np.ndarray,
                      point_to_reach: np.ndarray,
                      perturbation_shoulder: float | None = None,
                      perturbation_elbow: float | None = None,
                      scale_pm: float = 2.0,
                      scale_s1: float = 1.0,
                      t_init: float = 50.,
                      t_movement: float = 400.,
                      t_wait: float = 50.,
                      arms_model: PlanarArms | None = None):
    # make inputs
    input_pm_init, input_s1_init, _, _ = make_inputs(current_angles=current_thetas,
                                                     end_point=point_to_reach)

    if perturbation_shoulder is not None:
        current_thetas[0] += perturbation_shoulder
    if perturbation_elbow is not None:
        current_thetas[1] += perturbation_elbow

    current_thetas = PlanarArms.clip_values(current_thetas, radians=False)

    _, input_s1, _, new_thetas = make_inputs(current_angles=current_thetas,
                                             end_point=point_to_reach)

    trial(input_pm=input_pm_init * scale_pm,
          input_s1=input_s1_init * scale_s1,
          input_cm=None,
          t_wait=t_wait,
          t_sim=t_init,
          training=False,
          reset=False)

    _, output_theta = trial(input_pm=input_pm_init * scale_pm,
                            input_s1=input_s1 * scale_s1,
                            input_cm=None,
                            t_wait=0.,
                            t_sim=t_movement,
                            training=False)

    error = np.array((output_theta[0] - new_thetas[0], output_theta[1] - new_thetas[1]))
    output_theta = PlanarArms.clip_values(output_theta, radians=False)

    if arms_model is not None:
        arms_model.change_angle(arm='right', new_thetas=current_thetas, radians=False, num_iterations=int(t_init))
        arms_model.change_angle(arm='right', new_thetas=output_theta, radians=False, num_iterations=int(t_movement))

    return new_thetas, error
