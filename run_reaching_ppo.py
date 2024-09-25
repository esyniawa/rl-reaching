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
from utils import safe_save, generate_random_coordinate
from reward_functions import gaussian_reward, sigmoid_reward, logarithmic_reward

# torch.set_num_threads(4)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size: Tuple[int, int] = (128, 128)):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_size[0], 64),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(64, output_dim)
        self.actor_std = nn.Linear(64, output_dim)
        # make critic independent
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        shared_output = self.shared(x)
        mean = self.actor_mean(shared_output)
        std = nn.functional.softplus(
            self.actor_std(shared_output)) + 1e-6  # Softplus activation, adding a small value for numerical stability
        value = self.critic(shared_output)
        return mean, std, value


class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10, batch_size=64):
        self.network = PPONetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            mean, std, _ = self.network(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).numpy(), log_prob.squeeze(0).numpy()

    def update(self, replay_buffer: ReplayBuffer):
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
                _, _, next_values = self.network(next_states)
                returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
                _, _, values = self.network(states)
                advantages = returns - values.squeeze()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            mean, std, values = self.network(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = torch.exp(new_log_probs - old_log_probs.unsqueeze(1))

            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.unsqueeze(1)

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns.unsqueeze(1))
            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Save and load methods remain the same
    def save(self, path: str):
        if path[-1] != '/':
            path += '/'

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.network.state_dict(), path + 'network.pt')

    def load(self, path: str):
        if path[-1] != '/':
            path += '/'

        self.network.load_state_dict(torch.load(path + 'network.pt'))


# Environment wrapper
# TODO: implement init thetas in environment
class ReachingEnvironment:
    def __init__(self, init_thetas: np.ndarray = np.radians((90, 90)), radians: bool = True):

        if not radians:
            self.init_thetas = np.radians(init_thetas)
        else:
            self.init_thetas = init_thetas

        self.current_thetas = self.init_thetas.copy()
        self.target_thetas, self.target_pos = None, None
        self.max_distance = PlanarArms.upper_arm_length + PlanarArms.forearm_length

    def random_target(self):
        self.init_thetas = self.current_thetas.copy()
        # This function takes angles in degrees due to the other model!!!
        self.target_thetas, self.target_pos = generate_random_coordinate(init_thetas=self.init_thetas,
                                                                         normalize_xy=True,
                                                                         return_thetas_radians=True)

    def reset(self):
        return np.concatenate([self.init_thetas, self.target_pos])

    def set_thetas(self, thetas: np.ndarray):
        self.init_thetas = thetas
        self.current_thetas = thetas

    def step(self, action: np.ndarray, max_angle_change: float = np.radians(10), clip_thetas: bool = True):
        delta_thetas = np.clip(action, -max_angle_change, max_angle_change)  # prevent very large angle changes

        # clip angles to joint constraints
        new_thetas = self.current_thetas + delta_thetas
        if clip_thetas:
            new_thetas = PlanarArms.clip_values(new_thetas, radians=True)

        # Calculate new position
        new_pos = PlanarArms.forward_kinematics('right', new_thetas, radians=True, check_limits=False)[:, -1]

        # Calculate reward
        distance = np.linalg.norm(new_pos - self.target_pos)
        reward = 1.0 - (distance / self.max_distance)
        # reward smoother actions
        reward += 0.1 * (1.0 - np.sum(np.abs(action)) / (2 * max_angle_change))
        done = distance < 10  # 10mm threshold

        self.current_thetas = new_thetas
        return np.concatenate([new_thetas, self.target_pos]), reward, done


def collect_experience(args):
    env, agent, num_steps = args

    state = env.reset()
    experiences = []

    for _ in range(num_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, done = env.step(action)
        experiences.append((state, action, reward, next_state, done, log_prob))
        state = next_state
        if done:
            break

    return experiences


# Training loop
def train_ppo(Agent: PPOAgent,
              ReachEnv: ReachingEnvironment,
              num_reaching_trials: int,
              num_workers: int = 4,
              buffer_capacity: int = 100_000,
              steps_per_worker: int = 200,
              update_interval: int = 1000) -> PPOAgent:

    replay_buffer = ReplayBuffer(buffer_capacity)
    pool = Pool(num_workers)

    for trial in range(num_reaching_trials):
        # Initialize target for the environment
        ReachEnv.random_target()

        # Collect experiences using multiple workers
        worker_args = [(ReachEnv, Agent, steps_per_worker) for _ in range(num_workers)]
        worker_experiences = pool.map(collect_experience, worker_args)

        # Add experiences to the replay buffer
        for experiences in worker_experiences:
            for exp in experiences:
                replay_buffer.push(*exp)

        # TODO: Update the agent
        if (trial + 1) % update_interval == 0:
            Agent.update(replay_buffer)

    return Agent


def test_ppo(Agent: PPOAgent,
             ReachEnv: ReachingEnvironment,
             num_reaching_trials: int,
             init_thetas: np.ndarray = np.radians((90, 90)),
             max_steps: int = 200,  # beware the actions are clipped
             ) -> dict:

    ReachEnv.set_thetas(init_thetas)
    test_results = {
        'targets_thetas': [],
        'executed_thetas': [],
        'total_reward': [],
    }

    for trial in range(num_reaching_trials):
        # initialize target for the environment
        ReachEnv.random_target()
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

    return test_results


if __name__ == "__main__":
    sim_args_parser = argparse.ArgumentParser()
    sim_args_parser.add_argument('--id', type=int, default=0, help='Simulation ID')
    sim_args_parser.add_argument('--save', type=bool, default=True)
    sim_args_parser.add_argument('--do_plot', type=bool, default=True)
    sim_args_parser.add_argument('--num_workers', type=int, default=8)
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
    training_trials = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
    test_trials = 100

    # initialize agent
    state_dim = 4  # Current joint angles (2) + target position (2)
    action_dim = 2  # Changes in joint angles
    env = ReachingEnvironment()
    agent = PPOAgent(input_dim=state_dim, output_dim=action_dim)

    # training loop TODO: make ajustments
    for trials in training_trials:
        print(f'Sim {sim_args.id}: Training for {trials}...')
        subfolder = f'model_{trials}/'
        agent = train_ppo(agent, env,
                          num_reaching_trials=trials,
                          num_workers=sim_args.num_workers)

        if sim_args.save:
            agent.save(save_path_training + subfolder)

        print(f'Sim {sim_args.id}: Testing...')
        results_dict = test_ppo(agent, env, num_reaching_trials=test_trials, init_thetas=np.radians((90, 90)))
        if not os.path.exists(save_path_testing + subfolder):
            os.makedirs(save_path_testing + subfolder)
        np.savez(save_path_testing + subfolder + 'results.npz', **results_dict)

        if sim_args.do_plot:
            errors = []
            for target, exec in zip(results_dict['targets_thetas'], results_dict['executed_thetas']):
                error = PlanarArms.forward_kinematics(arm='right', thetas=target)[:, -1] - PlanarArms.forward_kinematics(arm='right', thetas=exec)[:, -1]
                errors.append(error)

            fig, axs = plt.subplots(nrows=2)
            axs[0].plot(np.array(errors))
            axs[0].set_title('Error')
            axs[1].plot(np.array(results_dict['total_reward']))
            axs[1].set_title('Reward')
            plt.savefig(save_path_testing + subfolder + 'error.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    print('Done!')
