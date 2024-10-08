import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta, Normal
import numpy as np
import argparse
from typing import Tuple, List
import warnings
from multiprocessing import Pool
from collections import deque
import random

from kinematics.planar_arms import PlanarArms
from utils import safe_save, generate_random_coordinate, norm_xy
from reward_functions_ppo import gaussian_reward, linear_reward, sigmoid_reward, logarithmic_reward

# torch.set_num_threads(4)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size=128):
        super(ActorNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, 64),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, output_dim)
        self.actor_std = nn.Linear(64, output_dim)

    def forward(self, x):
        shared_output = self.shared(x)
        mean = self.actor_mean(shared_output)
        std = nn.functional.softplus(self.actor_std(shared_output)) + 1e-6
        return mean, std


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layer_size=128):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, 1)
        )

    def forward(self, x):
        return self.critic(x)


class PPOAgent:
    def __init__(self,
                 input_dim,
                 output_dim,
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, epsilon=0.2,
                 epochs=10, batch_size=128):

        self.actor = ActorNetwork(input_dim, output_dim)
        self.critic = CriticNetwork(input_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def get_action(self, state, seed: int | None = None, add_exploration_noise: float | None = None):
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            state = torch.FloatTensor(state)
            mean, std = self.actor(state)

            # add noise to std to encourage or discourage exploration
            if add_exploration_noise is not None:
                std *= torch.abs(torch.randn_like(std) * add_exploration_noise)

            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).numpy(), log_prob.squeeze(0).numpy()

    def update(self, replay_buffer: ExperienceBuffer):
        if len(replay_buffer) < self.batch_size:
            return

        for _ in range(self.epochs):
            batch = replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, old_log_probs = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            old_log_probs = torch.FloatTensor(np.array(old_log_probs))
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            # Compute returns and advantages
            with torch.no_grad():
                next_values = self.critic(next_states)
                returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
                values = self.critic(states)
                advantages = returns - values.squeeze()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Actor loss
            mean, std = self.actor(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = torch.exp(new_log_probs - old_log_probs.unsqueeze(1))

            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.unsqueeze(1)

            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(self.critic(states), returns.unsqueeze(1))

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # clear buffer after each update because we are on-policy
        replay_buffer.clear()

    def save(self, path: str):
        if path[-1] != '/':
            path += '/'

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.actor.state_dict(), path + 'actor.pt')
        torch.save(self.critic.state_dict(), path + 'critic.pt')

    def load(self, path: str):
        if path[-1] != '/':
            path += '/'

        self.actor.load_state_dict(torch.load(path + 'actor.pt'))
        self.critic.load_state_dict(torch.load(path + 'critic.pt'))


# Environment wrapper
class ReachingEnvironment:
    def __init__(self,
                 target_thetas: np.ndarray,
                 init_thetas: np.ndarray,
                 target_pos: np.ndarray | None = None,
                 arm: str = 'right'):
        """

        :param target_thetas: target angles in radians
        :param init_thetas: initial angles in radians
        :param arm: left or right arm
        """

        self.arm = arm
        self.init_thetas = init_thetas
        self.current_thetas = init_thetas.copy()
        self.target_thetas = target_thetas
        if target_pos is None:
            self.target_pos = PlanarArms.forward_kinematics(arm=self.arm, thetas=self.target_thetas, radians=True)[:, -1]
        else:
            self.target_pos = target_pos

        # normalize target position to [0, 1]
        self.norm_target_pos = norm_xy(self.target_pos)

    @staticmethod
    def random_target(init_thetas: np.ndarray):
        target_thetas, target_pos = generate_random_coordinate(init_thetas=init_thetas,
                                                               return_thetas_radians=True)

        return target_thetas, target_pos

    def reset(self):
        self.current_thetas = self.init_thetas.copy()
        return np.concatenate([self.current_thetas, self.norm_target_pos])

    def set_parameters(self,
                       target_thetas: np.ndarray,
                       target_pos: np.ndarray,
                       init_thetas: np.ndarray):

        self.init_thetas, self.current_thetas = init_thetas, init_thetas
        self.target_thetas, self.target_pos = target_thetas, target_pos
        self.norm_target_pos = norm_xy(target_pos)

    def step(self,
             action: np.ndarray,
             abort_criteria: float = 1,  # in [mm]
             max_angle_change: float = np.radians(5),
             reward_gaussian: bool = True,
             clip_thetas: bool = True):

        # clip angles to joint constraints
        self.current_thetas += action * max_angle_change
        if clip_thetas:
            self.current_thetas = PlanarArms.clip_values(self.current_thetas, radians=True)

        # Calculate new position
        new_pos = PlanarArms.forward_kinematics(self.arm, self.current_thetas, radians=True, check_limits=False)[:, -1]

        # Calculate reward
        distance = np.linalg.norm(new_pos - self.target_pos)
        # TODO: Try different reward functions
        reward = linear_reward(error=distance, max_distance=PlanarArms.upper_arm_length + PlanarArms.forearm_length)
        if reward_gaussian:
            reward += gaussian_reward(error=distance, sigma=15)
        done = distance < abort_criteria

        return np.concatenate([self.current_thetas, self.norm_target_pos]), reward, done


def collect_experience(args):
    Agent, num_steps, init_thetas, target_thetas, target_pos = args

    # Set the seed for this worker
    # seed = torch.seed()

    env = ReachingEnvironment(target_thetas=target_thetas, target_pos=target_pos, init_thetas=init_thetas)
    state = env.reset()
    experiences = []

    for _ in range(num_steps):
        action, log_prob = Agent.get_action(state, add_exploration_noise=None, seed=None)
        next_state, reward, done = env.step(action)
        experiences.append((state, action, reward, next_state, done, log_prob))
        state = next_state
        if done:
            break

    del env
    return experiences


# Training loop
def train_ppo(Agent: PPOAgent,
              num_reaching_trials: int,
              num_workers: int = 10,
              buffer_capacity: int = 2000,
              steps_per_worker: int = 200,
              num_updates: int = 2,
              init_thetas: np.ndarray = np.radians((90, 90))) -> PPOAgent:

    replay_buffer = ExperienceBuffer(buffer_capacity)
    pool = Pool(num_workers)

    for trial in range(num_reaching_trials):
        # Initialize target for the environment
        target_thetas, target_pos = ReachingEnvironment.random_target(init_thetas=init_thetas)

        # Collect experiences using multiple workers
        worker_args = [(Agent, steps_per_worker, init_thetas, target_thetas, target_pos) for _ in range(num_workers)]
        worker_experiences = pool.map(collect_experience, worker_args)

        # Add experiences to the replay buffer
        for _ in range(num_updates):
            for experiences in worker_experiences:
                for exp in experiences:
                    replay_buffer.push(*exp)
            Agent.update(replay_buffer)

        # New initial angles are the current target angles
        init_thetas = target_thetas

    del pool

    return Agent


def test_ppo(Agent: PPOAgent,
             num_reaching_trials: int,
             init_thetas: np.ndarray = np.radians((90, 90)),
             max_steps: int = 200,  # beware the actions are clipped
             ) -> dict:

    target_thetas, target_pos = ReachingEnvironment.random_target(init_thetas=init_thetas)
    ReachEnv = ReachingEnvironment(init_thetas=init_thetas, target_thetas=target_thetas, target_pos=target_pos)

    test_results = {
        'targets_thetas': [],
        'executed_thetas': [],
        'total_reward': [],
    }

    for trial in range(num_reaching_trials):
        # initialize target for the environment
        state = ReachEnv.reset()
        episode_reward = 0

        for step in range(max_steps):
            action, _ = Agent.get_action(state)
            next_state, reward, done = ReachEnv.step(action)

            state = next_state
            episode_reward += reward

            if done:
                break

        test_results['targets_thetas'].append(ReachEnv.target_thetas)
        test_results['executed_thetas'].append(ReachEnv.current_thetas)
        test_results['total_reward'].append(episode_reward)

        # set new targets
        init_thetas = target_thetas
        target_thetas, target_pos = ReachingEnvironment.random_target(init_thetas=init_thetas)
        ReachEnv.set_parameters(target_thetas=target_thetas, target_pos=target_pos, init_thetas=init_thetas)

    return test_results


if __name__ == "__main__":
    sim_args_parser = argparse.ArgumentParser()
    sim_args_parser.add_argument('--id', type=int, default=0, help='Simulation ID')
    sim_args_parser.add_argument('--save', type=bool, default=True)
    sim_args_parser.add_argument('--do_plot', type=bool, default=True)
    sim_args_parser.add_argument('--num_workers', type=int, default=10)
    sim_args_parser.add_argument('--num_testing_trials', type=int, default=100)
    sim_args = sim_args_parser.parse_args()

    # import matplotlib if the error should be plotted
    if sim_args.do_plot:
        import matplotlib.pyplot as plt

    # save path
    save_path_training = f'results/training_ppo_{sim_args.id}/'
    save_path_testing = f'results/test_ppo_{sim_args.id}/'
    for path in [save_path_training, save_path_testing]:
        if not os.path.exists(path):
            os.makedirs(path)

    # parameters
    training_trials = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 52_000)
    test_trials = sim_args.num_testing_trials

    # initialize agent
    state_dim = 4  # Current joint angles (2) + target position (2)
    action_dim = 2  # Changes in joint angles
    agent = PPOAgent(input_dim=state_dim, output_dim=action_dim)

    # training loop TODO: make ajustments
    for trials in training_trials:
        print(f'Sim {sim_args.id}: Training for {trials}...')
        subfolder = f'model_{trials}/'
        agent = train_ppo(agent,
                          num_reaching_trials=trials,
                          num_workers=sim_args.num_workers)

        if sim_args.save:
            agent.save(save_path_training + subfolder)

        print(f'Sim {sim_args.id}: Testing...')
        results_dict = test_ppo(agent, num_reaching_trials=test_trials, init_thetas=np.radians((90, 90)))
        if not os.path.exists(save_path_testing + subfolder):
            os.makedirs(save_path_testing + subfolder)
        np.savez(save_path_testing + subfolder + 'results.npz', **results_dict)

        errors = []
        for target, exec in zip(results_dict['targets_thetas'], results_dict['executed_thetas']):
            error = np.linalg.norm(PlanarArms.forward_kinematics(arm='right', thetas=target)[:, -1] -
                                   PlanarArms.forward_kinematics(arm='right', thetas=exec)[:, -1])
            errors.append(error)

        np.save(save_path_testing + subfolder + 'error.npy', np.array(errors))

        if sim_args.do_plot:
            fig, axs = plt.subplots(nrows=2)
            axs[0].plot(np.array(errors))
            axs[0].set_title('Error')
            axs[1].plot(np.array(results_dict['total_reward']))
            axs[1].set_title('Reward')
            plt.savefig(save_path_testing + subfolder + 'error.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    print('Done!')
