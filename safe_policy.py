import os
import functools
import torch
import tqdm
import wandb
import numpy as np

from utils.dataset import DSRL_dataset
from utils.utils import *
from utils.info import *
from Agents.SRPO import SRPO
from copy import deepcopy


def pcgrad(grads, grads2):
    g1 = grads
    g2 = grads2
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()
    if g12 < 0:
        return ((1 - g12 / g11) * g1 + (1 - g12 / g22) * g2) / 2
    else:
        return (g1 + g2) / 2


def train_policy_safe(args, env, score_model, data_loader, start_epoch=0):
    n_epochs = args.n_policy_epochs
    num_1 = args.pretrain_epochs
    num_2 = args.train_epochs
    best_reward = -1e10
    best_cost = 1e10
    """
    need early stop, the num_2 does not need to be too large
    """
    tqdm_epoch_1 = tqdm.trange(start_epoch, num_1)
    tqdm_epoch_2 = tqdm.trange(start_epoch, num_2)
    evaluation_inerval = 10
    env_info = return_reference_scores(args.env)
    print(env_info)
    EPISODE_LENGTH = env_info["episode_length"]
    prev_policy_net = deepcopy(score_model.policy).to(args.device)
    reward_max = 0
    l = args.cost_scale
    for epoch in tqdm_epoch_1:
        for _ in range(1000):
            data = data_loader.sample(args.policy_batchsize)
            loss2 = score_model.update_policy(data, mode="bc")

        if (epoch % evaluation_inerval == (evaluation_inerval - 1)) or epoch == 0:
            env_info = return_reference_scores(args.env)
            EPISODE_LENGTH = env_info["episode_length"]
            reward_per_episode, cost_per_episode = evaluate_cost(
                args,
                env,
                data_loader,
                score_model.policy,
                EPISODE_LENGTH,
                trej_num=100,
            )
            reward_per_episode, cost_per_episode = np.array(
                reward_per_episode
            ), np.array(cost_per_episode)
            reward_normalized = (reward_per_episode - env_info["min_reward"]) / (
                env_info["max_reward"] - env_info["min_reward"]
            )
            cost_normalized = (cost_per_episode) / (l + 0.1)
            mean_reward = np.mean(reward_normalized)
            mean_cost = np.mean(cost_normalized)
            args.run.log({"eval/reward": mean_reward}, step=epoch + 1)
            args.run.log({"eval/cost": mean_cost}, step=epoch + 1)
            print("Average Cost is", mean_cost)
            print("Average Reward is", mean_reward)
    """
    pretrain the model for reward performance elevation
    """

    args.qc_thres = (
        args.cost_scale
        * (1 - args.gamma**EPISODE_LENGTH)
        / (1 - args.gamma)
        / EPISODE_LENGTH
    )

    for epoch in tqdm_epoch_2:
        args.beta = 0.01 * np.sqrt(epoch)

        reward_score, cost_score = evaluate_cost(
            args, env, data_loader, score_model.policy, EPISODE_LENGTH, trej_num=100
        )
        reward_per_episode, cost_per_episode = np.array(reward_score), np.array(
            cost_score
        )
        reward_normalized = (reward_per_episode - env_info["min_reward"]) / (
            env_info["max_reward"] - env_info["min_reward"]
        )
        cost_normalized = (cost_per_episode) / (args.cost_scale + 0.1)
        mean_reward = np.mean(reward_normalized)
        mean_cost = np.mean(cost_normalized)

        if mean_cost < args.cost_limit and mean_reward > best_reward:
            best_cost = mean_cost
            best_reward = mean_reward

        data = data_loader.sample(args.policy_batchsize)
        score_model.update_critic(data, TD=True)

        print("Reward is", mean_reward)
        print("Cost is ", mean_cost)

        args.run.log({"eval/reward": mean_reward}, step=epoch + num_1 + 1)
        args.run.log({"eval/cost": mean_cost}, step=epoch + num_1 + 1)

        # expore
        if epoch <= 500:
            data = data_loader.sample(args.policy_batchsize)
            q = score_model.update_policy(data)
            args.run.log({"train/q": q}, step=epoch + num_1 + 1)

        elif cost_normalized <= args.cost_limit - args.slack_bound:
            print("No cost Violation, Optimize the reward")
            data = data_loader.sample(args.policy_batchsize)
            q = score_model.update_policy(data)
            args.run.log({"train/q": q}, step=epoch + num_1 + 1)

        elif cost_normalized <= args.cost_limit + args.slack_bound:
            print("Optimize the cost and reward")
            data = data_loader.sample(args.policy_batchsize)
            prev_policy_net.load_state_dict(score_model.policy.state_dict())
            prev_policy_net_data = get_flat_params_from(prev_policy_net)
            q = score_model.update_policy(data)
            grads1 = get_flat_params_from(score_model.policy.net) - prev_policy_net_data
            set_flat_params_to(score_model.policy.net, prev_policy_net_data)
            if args.use_lagrangian:
                qc = score_model.update_policy_cost_lagrangian(data)
            else:
                qc = score_model.update_policy_cost(data)
            grads2 = get_flat_params_from(score_model.policy.net) - prev_policy_net_data
            # # # Ca grads -> final_grad = cagrad(grads1, grads2)
            final_grad = pcgrad(grads1, grads2)
            set_flat_params_to(
                score_model.policy.net, prev_policy_net_data + final_grad
            )
            args.run.log({"train/qc": qc}, step=epoch + num_1 + 1)
            args.run.log({"train/q": q}, step=epoch + num_1 + 1)
        else:
            print("optimize the cost")
            data = data_loader.sample(args.policy_batchsize)
            if args.use_lagrangian:
                qc = score_model.update_policy_cost_lagrangian(data)
            else:
                qc = score_model.update_policy_cost(data)
            args.run.log({"train/qc": qc}, step=epoch + num_1 + 1)

        """
        evaluation
        """
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            reward_per_episode, cost_per_episode = evaluate_cost(args,env,data_loader,score_model.policy,EPISODE_LENGTH,trej_num=100)
            reward_per_episode, cost_per_episode = np.array(reward_per_episode), np.array(cost_per_episode)
            reward_normalized = (reward_per_episode-env_info['min_reward'])/(env_info['max_reward']-env_info['min_reward'])
            cost_normalized = (cost_per_episode)/(args.cost_scale + 0.1)
            mean_reward = np.mean(reward_normalized)
            mean_cost = np.mean(cost_normalized)
            args.run.log({"eval/reward": mean_reward}, step=epoch+num_1+1)
            args.run.log({"eval/cost": mean_cost}, step=epoch+num_1+1)
        """
        save the policy model if it is safe and better than before 
        save the best model so far
        """
        if args.save_model and mean_cost < args.cost_limit and mean_reward > reward_max:
            print("-"*30)
            print("saving best model")
            print("-"*30)
            reward_max = mean_reward
            torch.save(score_model.policy.state_dict(), os.path.join("./Safe_policy_models", str(args.expid), "Gaussian_policy_best.pth".format(epoch+1)))

    print("Best reward is", best_reward)
    print("Best cost is ", best_cost)


"""
utlimate evaluation
load the best model we have
"""


def final_evaluate(args, env, score_model, data_loader):
    env_info = return_reference_scores(args.env)
    EPISODE_LENGTH = env_info["episode_length"]
    model_path = os.path.join(
        "./Safe_policy_models", str(args.expid), "Gaussian_policy_best.pth"
    )
    l = args.cost_scale
    ckpt = torch.load(model_path, map_location=args.device)
    score_model.policy.load_state_dict(ckpt)
    reward_per_episode, cost_per_episode = evaluate_cost(
        args, env, data_loader, score_model.policy, EPISODE_LENGTH, trej_num=1000
    )
    reward_per_episode, cost_per_episode = np.array(reward_per_episode), np.array(
        cost_per_episode
    )
    reward_normalized = (reward_per_episode - env_info["min_reward"]) / (
        env_info["max_reward"] - env_info["min_reward"]
    )
    cost_normalized = (cost_per_episode - env_info["min_cost"]) / (l + 0.1)
    mean_reward = np.mean(reward_normalized)
    mean_cost = np.mean(cost_normalized)
    print("Final reward is", mean_reward)
    print("Final cost is ", mean_cost)


def critic(args):
    # create folder
    for dir in ["./Safe_policy_models"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # create experiment folder
    if not os.path.exists(os.path.join("./Safe_policy_models", str(args.expid))):
        os.makedirs(os.path.join("./Safe_policy_models", str(args.expid)))

    # wandb init
    run = wandb.init(project=args.env, name=str(args.expid))
    wandb.config.update(args)
    args.run = run

    # create env and dataset
    if "drive" in args.env:
        import gym
    else:
        import gymnasium as gym

    env = gym.make(args.env)
    if "Gymnasium" in args.env:
        env.reset(seed=args.seed)
    else:
        env.seed(args.seed)

    dataset = DSRL_dataset(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # init marginal prob std function and score model
    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, device=args.device, beta_1=20.0
    )
    args.marginal_prob_std_fn = marginal_prob_std_fn

    score_model = SRPO(
        input_dim=state_dim + action_dim,
        output_dim=action_dim,
        marginal_prob_std=marginal_prob_std_fn,
        args=args,
    ).to(args.device)
    score_model.q[0].to(args.device)
    score_model.qc[0].to(args.device)

    # load the actor and critic
    if args.actor_load_path is not None:
        print("loading actor...")
        ckpt = torch.load(args.actor_load_path, map_location=args.device)
        score_model.load_state_dict(
            {k: v for k, v in ckpt.items() if "diffusion_behavior" in k}, strict=False
        )
    else:
        assert False

    if args.critic_load_path is not None:
        print("loading critic...")
        ckpt_reward = torch.load(
            args.critic_load_path + "critic_reward_ckpt50.pth", map_location=args.device
        )
        ckpt_cost = torch.load(
            args.critic_load_path + "critic_cost_ckpt50.pth", map_location=args.device
        )
        score_model.q[0].load_state_dict(ckpt_reward)
        score_model.qc[0].load_state_dict(ckpt_cost)
    else:
        assert False

    # train the policy
    print("training safe policy")
    train_policy_safe(args, env, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()


if __name__ == "__main__":
    args = get_args()
    critic(args)
