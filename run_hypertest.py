import numpy as np
import pandas as pd

from experiments import training, test_reach
from network.params import parameters
from network.model import *
import optuna
from optuna.samplers import TPESampler
import os


def update_model_params(sigma_s1: float = 25.,
                        sigma_pm: float = 125.,
                        sigma_m1: float = 20.,
                        rpe_error: float = 30.,
                        lr: float = 1.0,
                        decay: float = 1.0,
                        ):
    parameters['rpe_motor'] = rpe_error  # in [mm]
    parameters['sig_s1'] = sigma_s1  # in [°]
    parameters['sig_pm'] = sigma_pm  # in [mm]
    parameters['sig_m1'] = sigma_m1  # in [°]

    PM_StrD1.learning_rate = lr
    PM_StrD1.decay_rate = decay

    return parameters

def define_parameter_bounds() -> dict[str, tuple[float, float]]:
    return {
        'sigma_s1': (10., 100.),
        'sigma_pm': (40., 350.),
        'sigma_m1': (10., 100.),
        'rpe_error': (5., 100.),
        'lr': (0.1, 3.),
        'decay': (0.1, 3.),
    }


def update_reaching_space(parameters: dict,
                          x_bounds: tuple[float, float],
                          y_bounds: tuple[float, float], ):
    from network.utils import create_state_space

    # for cartesian space in PM
    parameters['x_reaching_space_limits'] = x_bounds
    parameters['y_reaching_space_limits'] = y_bounds

    parameters['state_pm'] = create_state_space(
        x_bound=parameters['x_reaching_space_limits'],
        y_bound=parameters['y_reaching_space_limits'],
        step_size_x=parameters['x_step_size'],
        step_size_y=parameters['y_step_size']
    )

    parameters['dim_pm'] = parameters['state_pm'].shape[:-1]

    return parameters


def objective(trial: optuna.Trial,
              study_name: str,
              feedback: bool,
              n_training_trials: int = 8_000,
              n_test_trials: int = 250,
              reward_time: int = 150,
              reach_time: int = 150,
              ) -> float:
    """Modified objective function incorporating new error terms."""
    save_path = f'results/{study_name}/optuna_trials/trial_{trial.number}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    bounds = define_parameter_bounds()

    # Sample parameters including temperature
    params = {
        'sigma_s1': trial.suggest_float('sigma_s1', *bounds['sigma_s1']),
        'sigma_pm': trial.suggest_float('sigma_pm', *bounds['sigma_pm']),
        'sigma_m1': trial.suggest_float('sigma_m1', *bounds['sigma_m1']),
        'rpe_error': trial.suggest_float('rpe_error', *bounds['rpe_error']),
        'lr': trial.suggest_float('lr', *bounds['lr']),
    }

    try:
        # Reset weights and update parameters
        PM_StrD1.w = 0.0
        update_model_params(**params)

        # Training
        training(N_trials=n_training_trials,
                 init_angle=np.array([90., 90.]),
                 reward_time=reward_time,
                 save_path=save_path,
                 disable_transmission_during_training=not feedback,
                 pop_monitor=None,
                 con_monitor=None,
                 animate_populations=False,
                 plot_error=False,
                 save_synapses=True)

        results_dict = test_reach(init_angle=np.array([90., 90.]),
                                  movement_time=reach_time,
                                  save_path=save_path,
                                  pop_monitor=None,
                                  test_condition='random',
                                  plot_error=True,
                                  animate_populations=False,
                                  arms_model=None,
                                  num_random_points=n_test_trials)

        total_error = np.mean(results_dict['error'])

        # Save trial results with raw and weighted errors
        trial_results = {
            'trial_number': trial.number,
            'total_error': total_error,
            **params
        }

        pd.DataFrame([trial_results]).to_csv(
            os.path.join(save_path, 'trial_results.csv'),
            index=False
        )

        return total_error

    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")
        return float('inf')


def run_optimization(n_hyper_trials: int,
                     feedback: bool,
                     study_name: str) -> optuna.Study:

    storage = f"sqlite:///results/{study_name}/optuna_trials/trials.db"

    # Create study using Optuna's built-in storage
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,  # Allow resuming existing study
        direction="minimize",
        sampler=TPESampler()
    )

    # Create objective function with only required arguments
    from functools import partial
    objective_partial = partial(objective, study_name=study_name, feedback=feedback)

    # Run optimization sequentially
    study.optimize(
        objective_partial,
        n_trials=n_hyper_trials,
        n_jobs=1,  # Run sequentially
        gc_after_trial=True,
        show_progress_bar=True
    )

    print("\nOptimization completed!")
    print(f"Best trial MSE: {study.best_value}")
    print("Best parameters:", study.best_params)

    # Save best parametersargs.data_set
    best_params_df = pd.DataFrame([study.best_params])
    os.makedirs(f'results/{study_name}/optuna_trials/', exist_ok=True)
    best_params_df.to_csv(
        os.path.join(f'results/{study_name}/optuna_trials/', 'best_params.csv'),
        index=False
    )

    return study


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--sim_id', type=int, default=0, help='Simulation ID')
    parser.add_argument('--study_name', type=str, default="default_model")
    parser.add_argument('--feedback', type=bool, default=True, help='Whether VL -> M1 is active or not.')
    args = parser.parse_args()

    # smaller peripersonal reaching space for hyperparameter optimization
    parameters = update_reaching_space(parameters, x_bounds=(-150, 100), y_bounds=(50, 250))

    study_name = args.study_name + f'_sim_{args.sim_id}'

    # Compile ANNarchy once at the start
    compile_folder = f'annarchy/optuna_model_tuning/{study_name}/'
    if not os.path.exists(compile_folder):
        os.makedirs(compile_folder)
    ann.compile(directory=compile_folder, clean=True)

    study = run_optimization(n_hyper_trials=args.n_trials, study_name=study_name, feedback=args.feedback)
