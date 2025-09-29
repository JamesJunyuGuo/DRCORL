import functools
import os

import dsrl

import numpy as np
import torch
import tqdm

import wandb
import time
from utils.dataset import DSRL_dataset
from Agents.SRPO import SRPO
from utils.utils import *
from utils.info import *
from copy import deepcopy


def evaluate(args, env, dataset,policy,EPISODE_LENGTH,trej_num=16): 
    policy.eval()
    state = env.observation_space.sample()
    state = state.reshape((1,-1))
    state = torch.tensor(state).to(args.device).float()
    start = time.time()
    for _ in range(1000):
        action = policy.act(state)
    end = time.time()
    print(f"Elapsed time: {end - start}")
       

    

def final_evaluate(args,env, score_model, data_loader):
    env_info = return_reference_scores(args.env)
    EPISODE_LENGTH = env_info['episode_length']
    model_path = os.path.join("./Safe_policy_models", str(args.expid), "Gaussian_policy_best.pth")
    ckpt = torch.load(model_path, map_location=args.device)
    score_model.policy.load_state_dict(ckpt)
    evaluate(args,env,data_loader,score_model.policy,EPISODE_LENGTH,trej_num=1000)

    

def critic(args):
    for dir in ["./Safe_policy_models"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./Safe_policy_models", str(args.expid))):
        os.makedirs(os.path.join("./Safe_policy_models", str(args.expid)))
    if 'drive' in args.env:
        import gym
    else:
        import gymnasium as gym
    env = gym.make(args.env)
    if 'Gymnasium'in args.env:
        env.reset(seed= args.seed)
    else:
        env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=20.0)
    args.marginal_prob_std_fn = marginal_prob_std_fn

    score_model= SRPO(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)


    dataset = DSRL_dataset(args)
    final_evaluate(args, env, score_model, dataset)
    print("finished")
    

if __name__ == "__main__":
    args = get_args()
    critic(args)
    
    