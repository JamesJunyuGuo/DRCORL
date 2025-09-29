import argparse
import dsrl
import gym
import numpy as np
import torch
import math
import random
import torch.nn.functional as F
from numba import njit
from numba.typed import List
from collections import namedtuple
from tqdm.auto import trange  # noqa


# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done','cost'))


def env_list():
    env_list = [# safety_gymnasium: car
            "OfflineCarButton1Gymnasium-v0",           # 0
            "OfflineCarButton2Gymnasium-v0",           # 1
            "OfflineCarCircle1Gymnasium-v0",           # 2
            "OfflineCarCircle2Gymnasium-v0",           # 3
            "OfflineCarGoal1Gymnasium-v0",             # 4
            "OfflineCarGoal2Gymnasium-v0",             # 5
            "OfflineCarPush1Gymnasium-v0",             # 6
            "OfflineCarPush2Gymnasium-v0",             # 7
            # safety_gymnasium: point
            "OfflinePointButton1Gymnasium-v0",         # 8
            "OfflinePointButton2Gymnasium-v0",         # 9
            "OfflinePointCircle1Gymnasium-v0",         # 10
            "OfflinePointCircle2Gymnasium-v0",         # 11
            "OfflinePointGoal1Gymnasium-v0",           # 12
            "OfflinePointGoal2Gymnasium-v0",           # 13
            "OfflinePointPush1Gymnasium-v0",           # 14
            "OfflinePointPush2Gymnasium-v0",           # 15
            # safety_gymnasium: velocity
            'OfflineAntVelocityGymnasium-v1',          # 16
            'OfflineHalfCheetahVelocityGymnasium-v1',  # 17
            'OfflineHopperVelocityGymnasium-v1',       # 18
            'OfflineSwimmerVelocityGymnasium-v1',      # 19
            'OfflineWalker2dVelocityGymnasium-v1',     # 20
            # bullet_safety_gym
            "OfflineCarCircle-v0",                     # 21
            "OfflineAntRun-v0",                        # 22
            "OfflineDroneRun-v0",                      # 23
            "OfflineDroneCircle-v0",                   # 24
            "OfflineCarRun-v0",                        # 25
            "OfflineAntCircle-v0",                     # 26
            "OfflineBallCircle-v0",                    # 27
            "OfflineBallRun-v0",                       # 28
            # point_robot
            "PointRobot",                              # 29
            ]
    return env_list



def track_env(env_name):
    env_list = [# safety_gymnasium: car
            "OfflineCarButton1Gymnasium-v0",           # 0
            "OfflineCarButton2Gymnasium-v0",           # 1
            "OfflineCarCircle1Gymnasium-v0",           # 2
            "OfflineCarCircle2Gymnasium-v0",           # 3
            "OfflineCarGoal1Gymnasium-v0",             # 4
            "OfflineCarGoal2Gymnasium-v0",             # 5
            "OfflineCarPush1Gymnasium-v0",             # 6
            "OfflineCarPush2Gymnasium-v0",             # 7
            # safety_gymnasium: point
            "OfflinePointButton1Gymnasium-v0",         # 8
            "OfflinePointButton2Gymnasium-v0",         # 9
            "OfflinePointCircle1Gymnasium-v0",         # 10
            "OfflinePointCircle2Gymnasium-v0",         # 11
            "OfflinePointGoal1Gymnasium-v0",           # 12
            "OfflinePointGoal2Gymnasium-v0",           # 13
            "OfflinePointPush1Gymnasium-v0",           # 14
            "OfflinePointPush2Gymnasium-v0",           # 15
            # safety_gymnasium: velocity
            'OfflineAntVelocityGymnasium-v1',          # 16
            'OfflineHalfCheetahVelocityGymnasium-v1',  # 17
            'OfflineHopperVelocityGymnasium-v1',       # 18
            'OfflineSwimmerVelocityGymnasium-v1',      # 19
            'OfflineWalker2dVelocityGymnasium-v1',     # 20
            # bullet_safety_gym
            "OfflineCarCircle-v0",                     # 21
            "OfflineAntRun-v0",                        # 22
            "OfflineDroneRun-v0",                      # 23
            "OfflineDroneCircle-v0",                   # 24
            "OfflineCarRun-v0",                        # 25
            "OfflineAntCircle-v0",                     # 26
            "OfflineBallCircle-v0",                    # 27
            "OfflineBallRun-v0",                       # 28
            # point_robot
            "PointRobot",                              # 29
            ]
    if env_name in env_list:
        return env_name
    else:
        raise NameError
    


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)

def merge_data(data_1,data_2):
    merged_dict = {key: data_1[key] + data_2[key] for key in data_1}
    return merged_dict



def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad



def marginal_prob_std(t, device="cuda",beta_1=20.0,beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """    
    t = torch.tensor(t, device=device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str) 
    parser.add_argument("--mode", default='online', type=str)      
    parser.add_argument("--device", default="cuda", type=str)      
    parser.add_argument("--save_model", default=1, type=int)      
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.2)  
    parser.add_argument('--gamma', type=float, default=0.99)      
    parser.add_argument('--cost_limit', type=float, default=1.0)
    parser.add_argument('--cost_scale', type=float, default=30.0)
    parser.add_argument('--slack_bound', type=float, default=0.1)    
    parser.add_argument('--actor_load_path', type=str, default=None)
    parser.add_argument('--critic_load_path', type=str, default=None)
    parser.add_argument('--policy_batchsize', type=int, default=256)              
    parser.add_argument('--actor_blocks', type=int, default=3)     
    parser.add_argument('--z_noise', type=int, default=1)
    parser.add_argument('--WT', type=str, default="VDS")
    parser.add_argument('--q_layer', type=int, default=2)
    parser.add_argument('--exploration-iteration', type=int, default=40, metavar='G',
                        help='the epoch number of the first performance exploration (default: 40)')
    
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
    parser.add_argument('--n_policy_epochs', type=int, default=100)
    parser.add_argument('--policy_layer', type=int, default=None)
    parser.add_argument('--critic_load_epochs', type=int, default=150)
    parser.add_argument('--regq', type=int, default=0)
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                        help='damping (default: 1e-1)')
    parser.add_argument('--exps-epoch', type=int, default=500, metavar='G',
                        help='the epoch number of exps (default: 500)')
    parser.add_argument('--episode_len', type=int, default=300)
    parser.add_argument('--qc_thres', type=float, default=0.0)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--q_lr', type=float, default=3e-4)
    parser.add_argument('--v_lr', type=float, default=3e-4)
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--diffusion_behavior_lr', type=float, default=3e-4)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--train_epochs', type=int, default=2000)
    parser.add_argument('--inference_step', type=int, default=1000)
    parser.add_argument('--q_mode', type=str, default="twin")
    parser.add_argument('--qc_mode', type=str, default="twin")
    parser.add_argument('--pid', type=bool, default=False)
    parser.add_argument('--use_lagrangian', type=bool, default=False)
    parser.add_argument('--pr_data', type=str, default="./data/point_robot-expert-random-100k.hdf5")
    print("**************************")
    args = parser.parse_known_args()[0]
    # args.qc_thres = args.cost_scale * (1 - args.gamma**args.episode_len) / (
    #         1 - args.gamma) / args.episode_len
    if args.debug:
        args.actor_epoch =1
        args.critic_epoch =1
    if args.policy_layer is None:
        args.policy_layer = 4 if "maze" in args.env else 2
    if "maze" in args.env:
        args.regq = 1
    
    print(args)
    return args


def evaluate_cost(args, env, dataset, policy, EPISODE_LENGTH, trej_num=16): 
    EPISODE_COUNT = trej_num
    EPISODE_NUM = 0
    num_steps = 0 
    num_episodes = 0
    cost_per_episode = []
    reward_per_episode = []
    while num_steps < EPISODE_LENGTH*EPISODE_COUNT:
        state, info = env.reset()
        cost_sum = 0
        reward_sum = 0
        for t in range(EPISODE_LENGTH): # Don't infinite loop while learning
            state = state.reshape((1,-1))
            action = policy.act(torch.tensor(state).to(args.device).float())
            action = action.detach().cpu().numpy().squeeze(0)
            next_state, reward, terminal, timeout, info = env.step(action)
            cost_sum += info["cost"]
            reward_sum += reward 
            if terminal or timeout:
                break
            state = next_state
        EPISODE_NUM += 1
        # print("Episode {} has ended".format(EPISODE_NUM))
        num_steps += EPISODE_LENGTH
        num_episodes += 1
        cost_per_episode.append(cost_sum)
        reward_per_episode.append(reward_sum)
        
        print("Evaluation Ended")
        return reward_per_episode, cost_per_episode
    
    else:
        raise ValueError("args does not match expected patterns ('online' or 'offline').")


def evaluate_reward(args, env, dataset,policy,EPISODE_LENGTH,trej_num=16): 
    EPISODE_COUNT = trej_num
    EPISODE_NUM = 0
    print("-"*30)
    print("Evaluation Begins")
    num_steps = 0 
    num_episodes = 0
    reward_per_episode = []
    while num_steps < EPISODE_LENGTH*EPISODE_COUNT:
        state, info = env.reset()
        reward_sum = 0
        for t in range(EPISODE_LENGTH): # Don't infinite loop while learning
            state = state.reshape((1,-1))
            action = policy.act(torch.tensor(state).to(args.device).float())
            action = action.detach().cpu().numpy().squeeze(0)
           
            # print("print action", action)
            # print(env.action_space.sample())
            next_state, reward, terminal, timeout, info = env.step(action)
            reward_sum += reward
            if terminal or timeout:
                break
            state = next_state
        EPISODE_NUM += 1
        # print("Episode {} has ended".format(EPISODE_NUM))
        num_steps += EPISODE_LENGTH
        num_episodes += 1
        reward_per_episode.append(reward_sum)
    print("-"*30)
    print("Evaluation Ends Now")
    return reward_per_episode
 
 
def offline_evaluate_cost(args, score_model,data,episode_length):
    states = data['s']
    actions = score_model.policy.act(states)
    qs = score_model.qc[0].q0_target.both(actions , states)
    value = ((qs[0]+qs[1])/2).mean().detach()
    cost_average_step = value*(1-args.gamma)
     
    return cost_average_step*episode_length
     
def evaluate_pr(args, env, policy, trej_num=16):

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []

    for _ in trange(trej_num, desc="Evaluating", leave=False):
        obs, info = env.reset()
        # print("obs", obs)
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            obs = obs.reshape((1,-1))
            action = policy.act(torch.tensor(obs).to(args.device).float())
            # print("action", action)
            action = action.detach().cpu().numpy().squeeze(0)
            # action,  = agent.eval_actions(obs)
            obs, reward, done, done, info = env.step(action)
            cost = info["violation"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return np.mean(episode_rets), np.mean(episode_costs)
     
    
def roll_out_trajectory(args, env, policy,MAX_LENGTH=1000, Trej_num = 10):
    states = []
    actions = []
    states_next = []
    rewards = []
    terminals = []
    costs = []
    state, info = env.reset()
    for _ in range(Trej_num):
        for t in range(MAX_LENGTH): # Don't infinite loop while learning
            state = state.reshape((1,-1))
            action = policy.act(torch.tensor(state).to(args.device).float())
            states.append(state.squeeze(0))
            action = action.detach().cpu().numpy().squeeze(0)
            actions.append(action)
            next_state, reward, terminal, timeout, info = env.step(action)
            states_next.append(next_state)
            rewards.append(reward)
            costs.append(info['cost'])
            terminals.append(terminal)
            if terminal or timeout:
                break
    states, actions, states_next, rewards, costs, terminals  = np.array(states), np.array(actions), np.array(states_next), np.array(rewards), np.array(costs), np.array(terminals)
    return {'observations':states,'actions': actions, 'next_observations':states_next, 'rewards':rewards, 'costs':costs, 'terminals': terminals }  

@njit
def compute_cost_reward_return(
    rew: np.ndarray,
    cost: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    returns,
    costs,
    starts,
    ends,
) -> np.ndarray:
    data_num = rew.shape[0]
    rew_ret, cost_ret = 0, 0
    is_start = True
    for i in range(data_num):
        if is_start:
            starts.append(i)
            is_start = False
        rew_ret += rew[i]
        cost_ret += cost[i]
        if terminals[i] or timeouts[i]:
            returns.append(rew_ret)
            costs.append(cost_ret)
            ends.append(i)
            is_start = True
            rew_ret, cost_ret = 0, 0

def get_trajectory_info(dataset: dict):
    # we need to initialize the numba List such that it knows the item type
    returns, costs = List([0.0]), List([0.0])
    # store the start and end indexes of the trajectory in the original data
    starts, ends = List([0]), List([0])
    data_num = dataset["rewards"].shape[0]
    print(f"Total number of data points: {data_num}")
    compute_cost_reward_return(
        dataset["rewards"], dataset["costs"], dataset["terminals"], dataset["timeouts"],
        returns, costs, starts, ends
    )
    return returns[1:], costs[1:], starts[1:], ends[1:]

class LagrangianPIDController:
    '''
    Lagrangian multiplier controller
    
    Args:
        KP (float): The proportional gain.
        KI (float): The integral gain.
        KD (float): The derivative gain.
        thres (float): The setpoint for the controller.
    '''

    def __init__(self, KP, KI, KD, thres) -> None:
        super().__init__()
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.thres = thres
        self.error_old = 0
        self.error_integral = 0

    def control(self, qc):
        '''
        @param qc [batch,]
        '''
        error_new = torch.mean(qc - self.thres)  # [batch]
        error_diff = F.relu(error_new - self.error_old)
        self.error_integral = torch.mean(F.relu(self.error_integral + error_new))
        self.error_old = error_new

        multiplier = F.relu(self.KP * F.relu(error_new) + self.KI * self.error_integral +
                            self.KD * error_diff)
        return torch.mean(multiplier) 
        
        
if __name__ == "__main__":
    args = get_args()
    print(args.qc_thres)
