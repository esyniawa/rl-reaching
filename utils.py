import numpy as np
import os
from network.params import parameters
from kinematics.planar_arms import PlanarArms


def generate_random_coordinate(theta_bounds_lower: float = parameters['theta_limit_low'],
                               theta_bounds_upper: float = parameters['theta_limit_high'],
                               x_bounds: tuple[float, float] = parameters['x_reaching_space_limits'],
                               y_bounds: tuple[float, float] = parameters['y_reaching_space_limits'],
                               clip_borders_theta: float = 10.,
                               clip_borders_xy: float = 10.,
                               init_thetas: np.ndarray | None = None,
                               return_thetas_radians: bool = False) -> tuple[np.ndarray, np.ndarray]:

    valid = False
    while not valid:
        random_thetas = np.random.uniform(low=theta_bounds_lower + clip_borders_theta,
                                          high=theta_bounds_upper - clip_borders_theta,
                                          size=2)

        random_xy = PlanarArms.forward_kinematics(arm=parameters['moving_arm'],
                                                  thetas=random_thetas,
                                                  radians=False)[:, -1]

        if (x_bounds[0] + clip_borders_xy < random_xy[0] < x_bounds[1] - clip_borders_xy
                and y_bounds[0] + clip_borders_xy < random_xy[1] < y_bounds[1] - clip_borders_xy):
            # check if thetas are far from each other
            if init_thetas is not None:
                # init thetas must be in degrees
                init_xy = PlanarArms.forward_kinematics(arm=parameters['moving_arm'],
                                                        thetas=init_thetas,
                                                        radians=False)[:, -1]
                if np.linalg.norm(init_xy - random_xy) > 50.0:
                    valid = True
            else:
                valid = True

    if return_thetas_radians:
        random_thetas = np.radians(random_thetas)

    return random_thetas, random_xy


def norm_xy(xy: np.ndarray,
            x_bounds: tuple[float, float] = parameters['x_reaching_space_limits'],
            y_bounds: tuple[float, float] = parameters['y_reaching_space_limits'],
            clip_borders_xy: float = 10.) -> np.ndarray:

    x_bounds = (x_bounds[0] + clip_borders_xy, x_bounds[1] - clip_borders_xy)
    y_bounds = (y_bounds[0] + clip_borders_xy, y_bounds[1] - clip_borders_xy)

    # Calculate the midpoints of x and y ranges
    x_mid = (x_bounds[0] + x_bounds[1]) / 2
    y_mid = (y_bounds[0] + y_bounds[1]) / 2

    # Calculate the half-ranges
    x_half_range = (x_bounds[1] - x_bounds[0]) / 2
    y_half_range = (y_bounds[1] - y_bounds[0]) / 2

    # Normalize to [-1, 1]
    normalized_x = (xy[0] - x_mid) / x_half_range
    normalized_y = (xy[1] - y_mid) / y_half_range

    return np.array([normalized_x, normalized_y])


def norm_distance(distance: np.ndarray,
                  x_bounds: tuple[float, float] = parameters['x_reaching_space_limits'],
                  y_bounds: tuple[float, float] = parameters['y_reaching_space_limits'],
                  clip_borders_xy: float = 10.
                  ) -> np.ndarray:

    x_bounds = (x_bounds[0] + clip_borders_xy, x_bounds[1] - clip_borders_xy)
    y_bounds = (y_bounds[0] + clip_borders_xy, y_bounds[1] - clip_borders_xy)

    # Calculate x and y ranges
    x_range = abs(x_bounds[1] - x_bounds[0])
    y_range = abs(y_bounds[1] - y_bounds[0])

    # Normalize the distance components to [-1, 1]
    normalized_dx = distance[0] / x_range
    normalized_dy = distance[1] / y_range

    return np.array([normalized_dx, normalized_dy])


def safe_save(save_name: str, array: np.ndarray) -> None:
    """
    If a folder is specified and does not yet exist, it will be created automatically.
    :param save_name: full path + data name
    :param array: array to save
    :return:
    """
    # create folder if not exists
    folder, data_name = os.path.split(save_name)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if data_name[-3:] == 'npy':
        np.save(save_name, array)
    else:
        np.save(save_name + '.npy', array)


def analyze_performance(test_results: dict, save_path: str | None = None, print_results: bool = True):
    """
    Analyze and visualize test results for DRL reaching task
    """
    import matplotlib.pyplot as plt


    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Error distribution
    axs[0, 0].hist(test_results['error'], bins=30)
    axs[0, 0].set_title('Error Distribution')
    axs[0, 0].set_xlabel('Error (mm)')
    axs[0, 0].set_ylabel('Count')

    # Steps to completion
    axs[0, 1].hist(test_results['steps'], bins=30)
    axs[0, 1].set_title('Steps to Completion')
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('Count')

    # Error over trials
    axs[1, 0].plot(test_results['error'])
    axs[1, 0].set_title('Error over Trials')
    axs[1, 0].set_xlabel('Trial')
    axs[1, 0].set_ylabel('Error (mm)')

    # Reward over trials
    axs[1, 1].plot(test_results['total_reward'])
    axs[1, 1].set_title('Reward over Trials')
    axs[1, 1].set_xlabel('Trial')
    axs[1, 1].set_ylabel('Total Reward')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    if print_results:
        # Print summary statistics
        print("\nPerformance Summary:")
        print(f"Success Rate: {test_results['success_rate']:.2f}%")
        print(f"Average Error: {np.mean(test_results['error']):.2f} ± {np.std(test_results['error']):.2f} mm")
        print(f"Average Steps: {np.mean(test_results['steps']):.2f} ± {np.std(test_results['steps']):.2f}")
        print(f"Average Reward: {np.mean(test_results['total_reward']):.2f} ± {np.std(test_results['total_reward']):.2f}")


if __name__ == '__main__':
    target_thetas, target_xy = generate_random_coordinate()
    current_thetas = np.radians((90., 90.))

    for _ in range(1000):
        current_thetas += np.random.normal(loc=0.0, scale=0.1, size=2)
        current_thetas = PlanarArms.clip_values(current_thetas, radians=True)
        current_pos = PlanarArms.forward_kinematics(arm=parameters['moving_arm'],
                                                    thetas=current_thetas,
                                                    radians=True)[:, -1]
        distance = target_xy - current_pos

        print(norm_distance(distance), distance)