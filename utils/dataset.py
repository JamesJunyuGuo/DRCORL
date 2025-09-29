import dsrl
# import gymnasium as gym
import gym
# import d4rl
import torch
import numpy as np 
from tqdm import trange
def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)

def cost_range(dataset, max_episode_steps):
    costs, lengths = [], []
    ep_cost, ep_len = 0., 0
    for c, d in zip(dataset['costs'], dataset['terminals']):
        ep_cost += float(c)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            costs.append(ep_cost)
            lengths.append(ep_len)
            ep_cost, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['costs'])
    return min(costs), max(costs)


class D4RL_dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        import d4rl
        self.args=args
        data = d4rl.qlearning_dataset(gym.make(args.env))
        self.device = args.device
        self.states = torch.from_numpy(data['observations']).float().to(self.device)
        self.actions = torch.from_numpy(data['actions']).float().to(self.device)
        self.next_states = torch.from_numpy(data['next_observations']).float().to(self.device)
        reward = torch.from_numpy(data['rewards']).reshape(-1, 1).float().to(self.device)
        self.is_finished = torch.from_numpy(data['terminals']).reshape(-1, 1).float().to(self.device)

        reward_tune = "iql_antmaze" if "antmaze" in args.env else "iql_locomotion"
        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            min_ret, max_ret = return_range(data, 1000)
            reward /= (max_ret - min_ret)
            reward *= 1000
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.rewards = reward
        print("data loaded")
        
        self.len = self.states.shape[0]
        self.current_idx = 0

    def __getitem__(self, index):
        use_index = index % self.len
        data = {'s': self.states[use_index],
                'a': self.actions[use_index],
                'r': self.rewards[use_index],
                's_':self.next_states[use_index],
                'd': self.is_finished[use_index],
            }
        return data

    def _shuffle_data(self):
        indices = torch.randperm(self.len).to("cuda")
        self.states = self.states[indices]
        self.next_states = self.next_states[indices]
        self.actions = self.actions[indices]
        self.rewards = self.rewards[indices]
        self.is_finished = self.is_finished[indices]

    def sample(self, batch_size):
        if self.current_idx+batch_size > self.len:
            self.current_idx = 0
        if self.current_idx == 0:
            self._shuffle_data()
        data = {'s': self.states[self.current_idx:self.current_idx+batch_size],
                'a': self.actions[self.current_idx:self.current_idx+batch_size],
                'r': self.rewards[self.current_idx:self.current_idx+batch_size],
                's_':self.next_states[self.current_idx:self.current_idx+batch_size],
                'd': self.is_finished[self.current_idx:self.current_idx+batch_size],
            }
        self.current_idx = self.current_idx + batch_size
        return data

    def __add__(self, online_data):
        pass
        

    def __len__(self):
        return self.len


class DSRL_dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args=args
        
        if 'drive' in args.env:
            import gym
        
        else:
            import gymnasium as gym
        
        self.env = gym.make(args.env)
        data = self.env.get_dataset()
        self.device = args.device
        self.states = torch.from_numpy(data['observations']).float().to(self.device)
        self.actions = torch.from_numpy(data['actions']).float().to(self.device)
        self.next_states = torch.from_numpy(data['next_observations']).float().to(self.device)
        self.costs = torch.from_numpy(data['costs']).float().to(self.device)
        reward = torch.from_numpy(data['rewards']).reshape(-1, 1).float().to(self.device)
        
        self.is_finished = torch.from_numpy(data['terminals']).reshape(-1, 1).float().to(self.device)

        reward_tune = "iql_antmaze" if "antmaze" in args.env else "iql_locomotion"
        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            min_ret, max_ret = return_range(data, 1000)
            reward /= (max_ret - min_ret)
            reward *= 1000
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.rewards = reward
        print("data loaded")
        
        self.len = self.states.shape[0]
        self.current_idx = 0

    def __getitem__(self, index):
        use_index = index % self.len
        data = {'s': self.states[use_index],
                'a': self.actions[use_index],
                'r': self.rewards[use_index],
                's_':self.next_states[use_index],
                'd': self.is_finished[use_index],
                'c': self.costs[use_index]
            }
        return data

    def _shuffle_data(self):
        indices = torch.randperm(self.len).to("cuda")
        self.states = self.states[indices]
        self.next_states = self.next_states[indices]
        self.actions = self.actions[indices]
        self.rewards = self.rewards[indices]
        self.is_finished = self.is_finished[indices]
        self.costs = self.costs[indices]

    def sample(self, batch_size):
        if self.current_idx+batch_size > self.len:
            self.current_idx = 0
        if self.current_idx == 0:
            self._shuffle_data()
        data = {'s': self.states[self.current_idx:self.current_idx+batch_size],
                'a': self.actions[self.current_idx:self.current_idx+batch_size],
                'r': self.rewards[self.current_idx:self.current_idx+batch_size],
                's_':self.next_states[self.current_idx:self.current_idx+batch_size],
                'd': self.is_finished[self.current_idx:self.current_idx+batch_size],
                'c': self.costs[self.current_idx:self.current_idx+batch_size],
            }
        self.current_idx = self.current_idx + batch_size
        return data

    def __add__(self, data_new):
        states_new = torch.from_numpy(data_new['observations']).float().to(self.device)
        actions_new = torch.from_numpy(data_new['actions']).float().to(self.device)
        next_states_new = torch.from_numpy(data_new['next_observations']).float().to(self.device)
        costs_new = torch.from_numpy(data_new['costs']).float().to(self.device)
        is_finished_new = torch.from_numpy(data_new['terminals']).float().to(self.device)
        reward_new = torch.from_numpy(data_new['rewards']).reshape(-1, 1).float().to(self.device)
        '''
        use the added data to renew the offline dataset with online data
        '''
        self.states = torch.cat((self.states, states_new), dim=0)
        self.actions = torch.cat((self.actions, actions_new), dim=0)
        self.rewards = torch.cat((self.rewards, reward_new), dim=0)
        self.next_states = torch.cat((self.next_states, next_states_new), dim=0)
        self.is_finished = torch.cat((self.is_finished, is_finished_new.view(-1, 1)), dim=0)
        self.costs = torch.cat((self.costs, costs_new), dim=0)

        
        

    def __len__(self):
        return self.len

from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'mask','cost'))

def transform_batch(batch):
    states = batch['s']           # Shape: (10, obs_dim)
    actions = batch['a']               # Shape: (10, act_dim)
    next_states = batch['s_'] # Shape: (10, obs_dim)
    rewards = batch['r']               # Shape: (10,)
    dones = batch['d']  
    costs = batch['c']

    transition_batch = Transition(
        state = states.detach().cpu().numpy(), 
        action = actions.detach().cpu().numpy(), 
        next_state = next_states.detach().cpu().numpy(), 
        reward = rewards.detach().cpu().numpy(), 
        mask = dones.detach().cpu().numpy(),
        cost = costs.detach().cpu().numpy(),
    )
    return transition_batch