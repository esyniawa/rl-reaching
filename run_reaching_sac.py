import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import argparse
from typing import Tuple, List
import warnings
from multiprocessing import Pool
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kinematics.planar_arms import PlanarArms
from utils import safe_save, generate_random_coordinate, norm_xy, norm_distance
from reward_functions import gaussian_reward, linear_reward, sigmoid_reward, logarithmic_reward

warnings.filterwarnings("ignore", category=DeprecationWarning)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    From "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al. (2013).
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GaussianActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size=128):
        super(GaussianActor, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_layer_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_layer_size, 64)),
            nn.ReLU()
        )

        # Mean and log_std heads
        self.mean = layer_init(nn.Linear(64, output_dim))
        self.log_std = layer_init(nn.Linear(64, output_dim))

        # Action rescaling
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent numerical instability
        return mean, log_std

    def sample(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Squash to [-1, 1]

        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds (github cleanrl)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action * self.action_scale + self.action_bias, log_prob


class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer_size=128):
        super(SoftQNetwork, self).__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim + action_dim, hidden_layer_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_layer_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1))
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class SACAgent:
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            alpha_lr: float = 3e-4,
            actor_lr: float = 3e-4,
            critic_lr: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            alpha: float = 0.2,
            automatic_entropy_tuning: bool = True,
            hidden_layer_size: int = 128
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Actor network
        self.actor = GaussianActor(input_dim, output_dim, hidden_layer_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks (two Q-functions to mitigate positive bias)
        self.critic1 = SoftQNetwork(input_dim, output_dim, hidden_layer_size)
        self.critic2 = SoftQNetwork(input_dim, output_dim, hidden_layer_size)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([output_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()

        # Action bounds
        self.max_action = 1
        self.min_action = -1

        # Device
        self.device = torch.device("cpu")
        self.actor = self.actor.to(self.device)
        self.critic1 = self.critic1.to(self.device)
        self.critic2 = self.critic2.to(self.device)
        self.critic1_target = self.critic1_target.to(self.device)
        self.critic2_target = self.critic2_target.to(self.device)

    def get_action(self, state, exploration_noise: bool = False) -> np.ndarray:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            if not exploration_noise:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        return action.squeeze(0).numpy()

    def update(self, replay_buffer, batch_size: int = 256):
        # Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)

            # Target Q-values
            next_q1 = self.critic1_target(next_state, next_action)
            next_q2 = self.critic2_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)

            # Soft Q-learning targets
            next_q = next_q - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * next_q

        # Current Q-values
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        new_action, log_prob = self.actor.sample(state)
        q1 = self.critic1(state, new_action)
        q2 = self.critic2(state, new_action)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_prob - q).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha if automatic entropy tuning is enabled
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        if path[-1] != '/':
            path += '/'

        torch.save(self.actor.state_dict(), path + 'actor.pt')
        torch.save(self.critic1.state_dict(), path + 'critic1.pt')
        torch.save(self.critic2.state_dict(), path + 'critic2.pt')
        if self.automatic_entropy_tuning:
            torch.save(self.log_alpha, path + 'log_alpha.pt')

    def load(self, path: str):
        if path[-1] != '/':
            path += '/'

        self.actor.load_state_dict(torch.load(path + 'actor.pt'))
        self.critic1.load_state_dict(torch.load(path + 'critic1.pt'))
        self.critic2.load_state_dict(torch.load(path + 'critic2.pt'))
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.load(path + 'log_alpha.pt')
            self.alpha = self.log_alpha.exp()

    def get_state_dict(self):
        """Get state dict for multiprocessing"""
        return {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None
        }

    def load_state_dict(self, state_dict):
        """Load state dict for multiprocessing"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        if self.automatic_entropy_tuning and state_dict['log_alpha'] is not None:
            self.log_alpha = state_dict['log_alpha']
            self.alpha = self.log_alpha.exp()

    @property
    def name(self):
        return 'sac'


# Environment wrapper
class ReachingEnvironment:
    def __init__(self,
                 target_thetas: np.ndarray,
                 init_thetas: np.ndarray,
                 target_pos: np.ndarray | None = None,
                 arm: str = 'right'):
        """

        :param target_thetas: target angles in radians
        :param target_pos: target position in cartesian coordinates [mm]. If None, it will be computed
        :param init_thetas: initial angles in radians
        :param arm: left or right arm
        """
        # parameters
        self.arm = arm
        self.init_thetas = init_thetas

        # current infos
        self.current_thetas = init_thetas.copy()
        self.current_pos = self.get_position(self.current_thetas)

        # target infos
        self.target_thetas = target_thetas
        if target_pos is None:
            self.target_pos = self.get_position(self.target_thetas)
        else:
            self.target_pos = target_pos

        # distance to target in normalized [-1, 1]
        self.norm_distance = norm_distance(self.target_pos - self.current_pos)

    @staticmethod
    def random_target(init_thetas: np.ndarray):
        target_thetas, target_pos = generate_random_coordinate(init_thetas=init_thetas,
                                                               return_thetas_radians=True)

        return target_thetas, target_pos

    def get_position(self, thetas: np.ndarray, radians: bool = True, check_bounds: bool = True) -> np.ndarray:
        return PlanarArms.forward_kinematics(arm=self.arm, thetas=thetas, radians=radians, check_limits=check_bounds)[:,
               -1]

    def reset(self):
        self.current_thetas = self.init_thetas.copy()
        self.current_pos = self.get_position(self.current_thetas)

        return np.concatenate([np.sin(self.current_thetas),
                               np.cos(self.current_thetas),
                               norm_distance(self.target_pos - self.current_pos)])

    def set_parameters(self,
                       target_thetas: np.ndarray,
                       target_pos: np.ndarray,
                       init_thetas: np.ndarray):

        self.__init__(arm=self.arm, target_thetas=target_thetas, init_thetas=init_thetas, target_pos=target_pos)

    def step(self,
             action: np.ndarray,
             abort_criteria: float = 2,  # in [mm]
             scale_angle_change: float = np.radians(5),
             reward_gaussian: bool = False,
             clip_thetas: bool = True,
             clip_penalty: bool = True):

        # Calculate new angles
        self.current_thetas += action * scale_angle_change
        reward = 0.
        # give penalty if action leads to out of joint bounds
        if clip_penalty:
            if self.current_thetas[0] < PlanarArms.l_upper_arm_limit or self.current_thetas[
                0] > PlanarArms.u_upper_arm_limit:
                reward -= 5.
            if self.current_thetas[1] < PlanarArms.l_forearm_limit or self.current_thetas[
                1] > PlanarArms.u_forearm_limit:
                reward -= 5.

        if clip_thetas:
            self.current_thetas = PlanarArms.clip_values(self.current_thetas, radians=True)

        # Calculate new position
        self.current_pos = self.get_position(self.current_thetas, check_bounds=clip_thetas)

        # Calculate error + reward
        distance = self.target_pos - self.current_pos
        self.norm_distance = norm_distance(distance)

        error = np.linalg.norm(distance)
        # TODO: Try different reward functions
        reward += -1e-3 * error  # in [m]
        if reward_gaussian:
            reward += gaussian_reward(error=error, sigma=10, amplitude=1.0)  # sigma in [mm]
        done = error < abort_criteria
        if done:
            reward += 10.

        return np.concatenate([np.sin(self.current_thetas),
                               np.cos(self.current_thetas),
                               self.norm_distance]), reward, done


def collect_experience_sac(args):
    agent_state_dict, num_steps, init_thetas, target_thetas, target_pos = args

    # Create a new agent instance for this worker
    worker_agent = SACAgent(input_dim=6, output_dim=2)  # Use same dimensions as main agent!! TODO: Make this less hacky
    worker_agent.load_state_dict(agent_state_dict)
    worker_agent.actor.eval()  # Set to eval mode

    # Set the seed for this worker
    torch.manual_seed(random.randint(0, 1000000))
    np.random.seed(random.randint(0, 1000000))

    env = ReachingEnvironment(target_thetas=target_thetas, target_pos=target_pos, init_thetas=init_thetas)
    state = env.reset()
    experiences = []

    for _ in range(num_steps):
        # Get action from worker agent
        with torch.no_grad():
            action = worker_agent.get_action(state, exploration_noise=True)
        next_state, reward, done = env.step(action, clip_thetas=True, clip_penalty=True)
        experiences.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            state = env.reset()

    del worker_agent  # Clean up
    return experiences


def train_sac_parallel(Agent: SACAgent,
                       num_reaching_trials: int,
                       replay_buffer: ReplayBuffer,
                       num_workers: int = 10,
                       steps_per_worker: int = 400,
                       batch_size: int = 128,
                       num_updates: int = 5,
                       init_thetas: np.ndarray = np.radians((90, 90))) -> SACAgent:
    with Pool(num_workers) as pool:
        for trial in range(num_reaching_trials):
            # Initialize target for all environments
            target_thetas, target_pos = ReachingEnvironment.random_target(init_thetas=init_thetas)

            # Get state dict for workers
            agent_state_dict = Agent.get_state_dict()

            # Collect experiences using multiple workers
            worker_args = [(agent_state_dict, steps_per_worker, init_thetas, target_thetas, target_pos)
                           for _ in range(num_workers)]
            worker_experiences = pool.map(collect_experience_sac, worker_args)

            # Add experiences to replay buffer
            for experiences in worker_experiences:
                for exp in experiences:
                    replay_buffer.push(*exp)

            # Perform multiple updates after collecting new experiences
            if len(replay_buffer) > batch_size:
                for _ in range(num_updates):
                    Agent.update(replay_buffer, batch_size)

            # New initial angles are the current target angles
            init_thetas = target_thetas

    return Agent


def test_sac(Agent: SACAgent,
             num_reaching_trials: int,
             init_thetas: np.ndarray = np.radians((90, 90)),
             max_steps: int = 400,
             render_interval: int | None = None,  # Set to N to render every N trials
             save_path: str | None = None  # Path to save renderings if render_interval is set
             ) -> dict:
    target_thetas, target_pos = ReachingEnvironment.random_target(init_thetas=init_thetas)
    ReachEnv = ReachingEnvironment(init_thetas=init_thetas, target_thetas=target_thetas, target_pos=target_pos)

    test_results = {
        'target_thetas': [],
        'executed_thetas': [],
        'target_pos': [],
        'error': [],
        'total_reward': [],
        'steps': [],
        'success_rate': 0.0  # Percentage of successful reaches
    }

    successful_reaches = 0

    for trial in range(num_reaching_trials):
        state = ReachEnv.reset()
        episode_reward = 0
        trajectory = []  # Store positions during this trial

        for step in range(max_steps):
            # Get deterministic action (no exploration noise during testing)
            action = Agent.get_action(state, exploration_noise=False)
            next_state, reward, done = ReachEnv.step(action)

            # Store current position for trajectory
            trajectory.append(ReachEnv.current_pos.copy())

            state = next_state
            episode_reward += reward

            if done:
                successful_reaches += 1
                break

        # Store results for this trial
        test_results['target_thetas'].append(ReachEnv.target_thetas.copy())
        test_results['executed_thetas'].append(ReachEnv.current_thetas.copy())
        test_results['target_pos'].append(ReachEnv.target_pos.copy())

        # Calculate final error
        final_error = np.linalg.norm(ReachEnv.target_pos - ReachEnv.current_pos)
        test_results['error'].append(final_error)
        test_results['total_reward'].append(episode_reward)
        test_results['steps'].append(step + 1)

        # Render if requested
        if render_interval and trial % render_interval == 0:
            render_trial(ReachEnv, trajectory,
                         save_path=f"{save_path}/trial_{trial}.png" if save_path else None)

        # Set new targets for next trial
        init_thetas = target_thetas
        target_thetas, target_pos = ReachingEnvironment.random_target(init_thetas=init_thetas)
        ReachEnv.set_parameters(target_thetas=target_thetas, target_pos=target_pos, init_thetas=init_thetas)

    # Calculate overall success rate
    test_results['success_rate'] = (successful_reaches / num_reaching_trials) * 100

    return test_results


def render_trial(env: ReachingEnvironment,
                 trajectory: np.ndarray | list,
                 save_path: str | None = None):
    """
    Render a single trial with trajectory
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(PlanarArms.x_limits)
    ax.set_ylim(PlanarArms.y_limits)
    ax.set_aspect('equal')
    ax.grid(True)

    # Plot trajectory
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3, label='Trajectory')

    # Plot final arm position
    arm_positions = PlanarArms.forward_kinematics(arm=env.arm,
                                                  thetas=env.current_thetas,
                                                  radians=True)
    ax.plot(arm_positions[0], arm_positions[1], 'bo-', lw=2, label='Final Position')

    # Plot target
    ax.plot(env.target_pos[0], env.target_pos[1], 'r*', markersize=10, label='Target')

    # Add legend
    ax.legend()

    # Add final error text
    final_error = np.linalg.norm(env.target_pos - env.current_pos)
    ax.text(0.02, 0.95, f'Final Error: {final_error:.2f} mm',
            transform=ax.transAxes)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def render_reaching(Agent: SACAgent,
                    init_thetas: np.ndarray = np.radians((90, 90)),
                    max_steps: int = 400,
                    fps: int = 50,
                    save_path: str | None = None):
    # Initialize environment
    target_thetas, target_pos = ReachingEnvironment.random_target(init_thetas=init_thetas)
    env = ReachingEnvironment(init_thetas=init_thetas, target_thetas=target_thetas, target_pos=target_pos)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(PlanarArms.x_limits)
    ax.set_ylim(PlanarArms.y_limits)
    ax.set_aspect('equal')
    ax.grid(True)

    # Initialize lines for arm segments and target
    line, = ax.plot([], [], 'bo-', lw=2)
    target, = ax.plot(target_pos[0], target_pos[1], 'r*', markersize=10, linestyle=None)

    # Text for step count and error
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    error_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        target.set_data([], [])
        step_text.set_text('')
        error_text.set_text('')
        return line, target, step_text, error_text

    def animate(i):
        done = False
        if done:
            return line, target, step_text, error_text
        else:
            if i == 0:
                global state
                state = env.reset()
            else:
                action = Agent.get_action(state, exploration_noise=False)
                state, _, done = env.step(action)

        # Get current arm position
        arm_positions = PlanarArms.forward_kinematics(arm=env.arm, thetas=env.current_thetas, radians=True)
        x_coords = list(arm_positions[0])
        y_coords = list(arm_positions[1])

        line.set_data(x_coords, y_coords)
        target.set_data(env.target_pos[0], env.target_pos[1])

        error = np.linalg.norm(env.target_pos - env.current_pos)
        step_text.set_text(f'Step: {i}')
        error_text.set_text(f'Error: {error:.2f} mm')

        return line, target, step_text, error_text

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=max_steps,
                                   interval=fps, blit=True)

    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=30)
    else:
        plt.show()

    plt.close(fig)


def analyze_performance(test_results: dict, save_path: str | None = None):
    """
    Analyze and visualize test results
    """
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

    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Success Rate: {test_results['success_rate']:.2f}%")
    print(f"Average Error: {np.mean(test_results['error']):.2f} ± {np.std(test_results['error']):.2f} mm")
    print(f"Average Steps: {np.mean(test_results['steps']):.2f} ± {np.std(test_results['steps']):.2f}")
    print(f"Average Reward: {np.mean(test_results['total_reward']):.2f} ± {np.std(test_results['total_reward']):.2f}")


if __name__ == "__main__":
    sim_args_parser = argparse.ArgumentParser()
    sim_args_parser.add_argument('--id', type=int, default=0, help='Simulation ID')
    sim_args_parser.add_argument('--save', type=bool, default=True)
    sim_args_parser.add_argument('--do_plot', type=bool, default=True)
    sim_args_parser.add_argument('--num_workers', type=int, default=10)
    sim_args_parser.add_argument('--num_testing_trials', type=int, default=100)
    sim_args_parser.add_argument('--buffer_size', type=int, default=100_000)
    sim_args_parser.add_argument('--batch_size', type=int, default=256)
    sim_args = sim_args_parser.parse_args()

    # import matplotlib if the error should be plotted
    if sim_args.do_plot:
        import matplotlib.pyplot as plt

    # save path
    save_path_training = f'results/training_sac_{sim_args.id}/'
    save_path_testing = f'results/test_sac_{sim_args.id}/'
    for path in [save_path_training, save_path_testing]:
        if not os.path.exists(path):
            os.makedirs(path)

    # parameters
    training_trials = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000,)
    test_trials = sim_args.num_testing_trials

    # initialize agent
    state_dim = 6  # Current joint angles (4) + cartesian error to target position (2)
    action_dim = 2  # Changes in joint angles
    agent = SACAgent(input_dim=state_dim, output_dim=action_dim)

    replay_buffer = ReplayBuffer(capacity=sim_args.buffer_size)
    # training loop TODO: make ajustments
    for trials in training_trials:
        print(f'Sim {sim_args.id}: Training for {trials}...')
        subfolder = f'model_{trials}/'
        if not os.path.exists(save_path_training + subfolder):
            os.makedirs(save_path_training + subfolder)

        agent = train_sac_parallel(agent,
                                   num_reaching_trials=trials,
                                   replay_buffer=replay_buffer,
                                   num_workers=sim_args.num_workers,
                                   batch_size=sim_args.batch_size)

        if sim_args.save:
            agent.save(save_path_training + subfolder)

        print(f'Sim {sim_args.id}: Testing...')
        results_dict = test_sac(agent,
                                num_reaching_trials=test_trials,
                                init_thetas=np.radians((90, 90)))

        if not os.path.exists(save_path_testing + subfolder):
            os.makedirs(save_path_testing + subfolder)
        np.savez(save_path_testing + subfolder + 'results.npz', **results_dict)

        if sim_args.do_plot:
            analyze_performance(results_dict, save_path=save_path_testing + subfolder + "performance.pdf")

    print('Done!')
