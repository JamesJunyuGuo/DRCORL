import os
import torch
import tqdm
import wandb
import functools
import numpy as np

from utils.dataset import DSRL_dataset
from Agents.SRPO import SRPO_Behavior
from utils.utils import get_args, marginal_prob_std


def train_behavior(args, score_model, data_loader, start_epoch=0):
    """Train the behavior model using diffusion-based approach.
    Args:
        args: Command line arguments.
        score_model: The behavior model to be trained.
        data_loader: The dataset loader.
        start_epoch: The starting epoch for training.
    """
    # Training parameters
    n_epochs = 200
    evaluation_interval = 5
    save_interval = 20
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)

    # Training loop
    for epoch in tqdm_epoch:
        avg_loss = 0.0
        num_items = 0

        # Each epoch consists of 10,000 iterations
        for _ in range(10000):
            data = data_loader.sample(2048)
            score_model.update_behavior(data)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1

        tqdm_epoch.set_description("Average Loss: {:5f}".format(avg_loss / num_items))

        # Evaluation and logging
        if (epoch % evaluation_interval == (evaluation_interval - 1)) or epoch == 0:
            args.run.log(
                {"loss/diffusion": score_model.loss.detach().cpu().numpy()},
                step=epoch + 1,
            )

        # Model checkpointing
        if args.save_model and (
            (epoch % save_interval == (save_interval - 1)) or epoch == 0
        ):
            torch.save(
                score_model.state_dict(),
                os.path.join(
                    "./Safe_model_factory",
                    str(args.expid),
                    "behavior_ckpt{}.pth".format(epoch + 1),
                ),
            )


def behavior(args):
    # create folder
    for dir in ["./Safe_model_factory"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # create experiment folder
    if not os.path.exists(os.path.join("./Safe_model_factory", str(args.expid))):
        os.makedirs(os.path.join("./Safe_model_factory", str(args.expid)))

    # wandb init
    run = wandb.init(project="Safe_model_factory", name=str(args.expid))
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

    # init marginal prob std function and score model
    marginal_prob_std_fn = functools.partial(
        marginal_prob_std, device=args.device, beta_1=20.0
    )
    args.marginal_prob_std_fn = marginal_prob_std_fn

    score_model = SRPO_Behavior(
        input_dim=state_dim + action_dim,
        output_dim=action_dim,
        marginal_prob_std=marginal_prob_std_fn,
        args=args,
    ).to(args.device)

    # start training
    print("training behavior")
    train_behavior(args, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()


if __name__ == "__main__":
    args = get_args()
    behavior(args)
