import numpy as np
from kinematics.planar_arms import PlanarArms
from .utils import create_state_space

parameters = {}

# kinematic parameters
parameters['moving_arm'] = 'right'

# for motor space
parameters['dim_motor'] = (22, 2)

parameters['theta_limit_low'], parameters['theta_limit_high'] = -10, 170
parameters['motor_orientations'] = np.linspace(-180, 180, parameters['dim_motor'][0], endpoint=True)

parameters['motor_step_size'] = abs(parameters['motor_orientations'][1] - parameters['motor_orientations'][0])

# for joint space in S1
parameters['shoulder_limits'] = np.array((parameters['theta_limit_low'], parameters['theta_limit_high']))
parameters['elbow_limits'] = np.array((parameters['theta_limit_low'], parameters['theta_limit_high']))

parameters['s1_step_size'] = 10.  # in [°]

# TODO: check intersection with cartesian space
parameters['state_s1'] = create_state_space(
    x_bound=parameters['shoulder_limits'],
    y_bound=parameters['elbow_limits'],
    step_size_x=parameters['s1_step_size'],
    step_size_y=parameters['s1_step_size']
)

parameters['dim_s1'] = parameters['state_s1'].shape[:-1]

# for cartesian space in PM
parameters['x_reaching_space_limits'] = (-300, 200)
parameters['y_reaching_space_limits'] = (0, 360)

parameters['x_step_size'] = 20.
parameters['y_step_size'] = 20.

parameters['state_pm'] = create_state_space(
    x_bound=parameters['x_reaching_space_limits'],
    y_bound=parameters['y_reaching_space_limits'],
    step_size_x=parameters['x_step_size'],
    step_size_y=parameters['y_step_size']
)

parameters['dim_pm'] = parameters['state_pm'].shape[:-1]

# model definitions
parameters['sig_s1'] = 40.  # in [°]
parameters['sig_pm'] = 160.  # in [mm]
parameters['sig_m1'] = 30.  # in [°]

parameters['dim_bg'] = parameters['dim_motor']

parameters['strength_m1'] = 0.3
parameters['strength_snr'] = 0.6

parameters['depth_str'] = parameters['dim_motor'][0] * 2
parameters['subsets_str'] = {
    'shoulder_like': (0, parameters['depth_str'] // 2),
    'elbow_like': (parameters['depth_str'] // 2, parameters['depth_str']),
}

parameters['dim_str'] = tuple(list(parameters['dim_s1']) + [parameters['depth_str']])

# Test setup
parameters['test_trajectory_cube'] = [
    np.array((-120, 260)),
    np.array((-120, 160)),
    np.array((120, 160)),
    np.array((120, 260)),
]

parameters['test_trajectory_circle'] = [
    np.array((-120, 260)),
    np.array((-120, 160)),
    np.array((120, 160)),
    np.array((120, 260)),
]

# dictionary of plotting types the different populations
parameters['pop_plot_types'] = {
    'PM': 'Matrix',
    'S1': 'Matrix',
    'StrD1': 'Matrix',
    'GPe': 'Plot',
    'SNr': 'Plot',
    'CM': 'Plot',
    'VL': 'Plot',
    'M1': 'Plot',
    'SNc': 'Line',
    'Output_Shoulder': 'Polar',
    'Output_Elbow': 'Polar',
}