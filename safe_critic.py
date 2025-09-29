import os
import torch
import tqdm
import wandb
import numpy as np

from Agents.SRPO import SRPO_IQL
from utils.dataset import DSRL_dataset
from utils.utils import get_args
from utils.utils import evaluate_cost, evaluate_pr
from utils.info import return_reference_scores


def train_critic(args, env, score_model, data_loader, start_epoch=0):
    """Train the critic model using IQL approach.
    Args:
        args: Command line arguments.
        env: The environment for evaluation.
        score_model: The critic model to be trained.
        data_loader: The dataset loader.
        start_epoch: The starting epoch for training.
    """
    # Training parameters
    n_epochs = 100
    evaluation_interval = 5
    save_interval = 10
    env_info = return_reference_scores(args.env)
    EPISODE_LENGTH = env_info["episode_length"]
    l = args.cost_scale
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)

    # Training loop
    for epoch in tqdm_epoch:
        avg_loss = 0.0
        num_items = 0

        # Each epoch consists of 10,000 iterations
        for _ in range(10000):
            data = data_loader.sample(256)
            loss2 = score_model.update_iql(data)

            avg_loss += 0.0
            num_items += 1

        tqdm_epoch.set_description("Average Loss: {:5f}".format(avg_loss / num_items))

        # Evaluation and logging
        if (epoch % evaluation_interval == (evaluation_interval - 1)) or epoch == 0:
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
            args.run.log(
                {"loss/reward_v_loss": score_model.q[0].v_loss.detach().cpu().numpy()},
                step=epoch + 1,
            )
            args.run.log(
                {"loss/reward_q_loss": score_model.q[0].q_loss.detach().cpu().numpy()},
                step=epoch + 1,
            )
            args.run.log(
                {"loss/reward_q": score_model.q[0].q.detach().cpu().numpy()},
                step=epoch + 1,
            )
            args.run.log(
                {"loss/reward_v": score_model.q[0].v.detach().cpu().numpy()},
                step=epoch + 1,
            )
            args.run.log(
                {"loss/cost_v_loss": score_model.qc[0].v_loss.detach().cpu().numpy()},
                step=epoch + 1,
            )
            args.run.log(
                {"loss/cost_q_loss": score_model.qc[0].q_loss.detach().cpu().numpy()},
                step=epoch + 1,
            )
            args.run.log(
                {"loss/cost_q": score_model.qc[0].q.detach().cpu().numpy()},
                step=epoch + 1,
            )
            args.run.log(
                {"loss/cost_v": score_model.qc[0].v.detach().cpu().numpy()},
                step=epoch + 1,
            )
            args.run.log(
                {"loss/policy_loss": score_model.policy_loss.detach().cpu().numpy()},
                step=epoch + 1,
            )

        # Model checkpointing
        if args.save_model and (
            (epoch % save_interval == (save_interval - 1)) or epoch == 0
        ):
            torch.save(
                score_model.q[0].state_dict(),
                os.path.join(
                    "./Safe_model_factory",
                    str(args.expid),
                    "critic_reward_ckpt{}.pth".format(epoch + 1),
                ),
            )
            torch.save(
                score_model.qc[0].state_dict(),
                os.path.join(
                    "./Safe_model_factory",
                    str(args.expid),
                    "critic_cost_ckpt{}.pth".format(epoch + 1),
                ),
            )


def critic(args):
    # create folder
    for dir in ["./Safe_model_factory"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # create experiment folder
    if not os.path.exists(os.path.join("./Safe_model_factory", str(args.expid))):
        os.makedirs(os.path.join("./Safe_model_factory", str(args.expid)))

    # wandb init
    run = wandb.init(project="Safemodel_factory", name=str(args.expid))
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

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # init model
    score_model = SRPO_IQL(
        input_dim=state_dim + action_dim, output_dim=action_dim, args=args
    ).to(args.device)
    score_model.q[0].to(args.device)
    score_model.qc[0].to(args.device)

    # start training
    print("training critic")
    train_critic(args, env, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()


if __name__ == "__main__":
    args = get_args()
    critic(args)
