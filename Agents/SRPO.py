# Safe Reinforcement Policy Optimization (SRPO) Agent Implementation
# This module implements SRPO, a safe reinforcement learning algorithm that uses
# diffusion models for policy optimization while incorporating safety constraints
# through cost-aware value functions and Lagrangian multipliers.

import copy
import torch
import torch.nn as nn

# Import custom models and utilities
from Agents.model import *
from utils.utils import LagrangianPIDController


class SRPO(nn.Module):
    """
    Safe Reinforcement Policy Optimization (SRPO) Agent
    
    SRPO combines diffusion models with safe reinforcement learning to optimize policies
    while respecting safety constraints. It uses:
    - A diffusion behavior model for score-based policy guidance
    - Twin/ensemble Q-networks for reward estimation 
    - Twin/ensemble Qc-networks for cost estimation
    - Lagrangian multipliers with PID control for constraint satisfaction
    
    Args:
        input_dim (int): Dimension of state-action space (state_dim + action_dim)
        output_dim (int): Dimension of action space
        marginal_prob_std (callable): Function for diffusion noise schedule
        PID (list): PID controller gains [KP, KI, KD] for Lagrangian multiplier
        args: Configuration arguments containing hyperparameters
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        marginal_prob_std,
        PID: list = [0.1, 0.003, 0.001],
        args=None,
    ):
        super().__init__()
        
        # Initialize diffusion behavior model for score-based guidance
        # This model learns to predict noise/score for denoising actions
        self.diffusion_behavior = ScoreNet_IDQL(
            input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args
        )
        self.diffusion_optimizer = torch.optim.AdamW(
            self.diffusion_behavior.parameters(), lr=args.diffusion_behavior_lr
        )
        
        # Initialize deterministic policy network (Dirac policy)
        # Maps states to deterministic actions
        self.policy = Dirac_Policy(
            output_dim, input_dim - output_dim, layer=args.policy_layer
        ).to("cuda")
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=args.policy_lr
        )
        # Cosine annealing learning rate scheduler for policy
        self.policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.0
        )
        
        # PID controller for Lagrangian multiplier adaptation
        self.KP, self.KI, self.KD = PID
        self.controller = LagrangianPIDController(
            KP=self.KP, KI=self.KI, KD=self.KD, thres=args.qc_thres
        )

        # Store configuration and dimensions
        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        
        # Initialize value function critics
        self.q = []  # Reward Q-function
        self.qc = [] # Cost Q-function (safety critic)
        
        # Add reward critic (twin or ensemble mode)
        self.q.append(
            IQL_Critic(
                adim=output_dim,
                sdim=input_dim - output_dim,
                args=args,
                mode=args.q_mode,
            )
        )
        # Add cost critic (twin or ensemble mode)
        self.qc.append(
            IQL_Critic(
                adim=output_dim,
                sdim=input_dim - output_dim,
                args=args,
                mode=args.qc_mode,
            )
        )

    def update_policy(self, data, mode="ql"):
        """
        Update policy using diffusion-based policy optimization.
        
        This method combines score matching loss with Q-function guidance to optimize
        the policy. The loss includes:
        1. Score matching term: aligns policy actions with diffusion model predictions
        2. Q-function guidance: encourages actions that maximize expected reward
        
        Args:
            data (dict): Batch of data containing states
            mode (str): "ql" for Q-learning guidance, "bc" for behavior cloning only
        
        Returns:
            Average Q-value for monitoring
        """
        s = data["s"]  # Extract states from data
        
        # Set diffusion model to evaluation mode for inference
        self.diffusion_behavior.eval()
        
        # Generate actions from current policy
        a = self.policy.act(s)
        
        # Sample random timesteps for diffusion process (avoid t=0 and t=1)
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        
        # Get noise schedule parameters at time t
        alpha_t, std = self.marginal_prob_std(t)
        
        # Sample noise and create perturbed actions
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        
        # Get score/noise prediction from diffusion model
        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()
            # Adjust prediction based on weighting type
            if "noise" in self.args.WT:
                episilon = episilon - z
        
        # Compute weighting term based on diffusion schedule
        if "VDS" in self.args.WT:  # Variance preserving SDE
            wt = std**2
        elif "stable" in self.args.WT:  # Stable weighting
            wt = 1.0
        elif "score" in self.args.WT:  # Score-based weighting
            wt = alpha_t / std
        else:
            assert False, "Unknown weighting type"
        
        # Compute Q-function guidance gradients
        detach_a = a.detach().requires_grad_(True)
        qs = self.q[0].q0_target.both(detach_a, s)  # Get twin Q-values
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0  # Average twin Q-values
        self.policy.q = torch.mean(q)  # Store for monitoring
        
        # Compute gradients of Q-function w.r.t. actions (policy guidance)
        guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        
        # Optional: normalize guidance to prevent exploding gradients
        if self.args.regq:
            guidance_norm = torch.mean(guidance**2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm
        
        # Compute loss based on mode
        if mode == "bc":
            # Behavior cloning: only score matching loss
            loss = (episilon * a).sum(-1) * wt
        else:
            # Q-learning: score matching + Q-function guidance
            loss = (episilon * a).sum(-1) * wt - (guidance * a).sum(-1) * self.args.beta
        
        # Optimize policy
        loss = loss.mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()
        
        # Return diffusion model to training mode
        self.diffusion_behavior.train()

        return self.policy.q

    def update_policy_cost(self, data):
        """
        Update policy with cost-based guidance (safety-aware optimization).
        
        Similar to update_policy but uses cost Q-function for guidance instead of
        reward Q-function. The policy is penalized for actions that lead to high costs.
        
        Args:
            data (dict): Batch of data containing states
            
        Returns:
            Average cost Q-value for monitoring
        """
        s = data["s"]
        self.diffusion_behavior.eval()
        
        # Generate actions and setup diffusion process (same as update_policy)
        a = self.policy.act(s)
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        
        # Get diffusion model predictions
        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()
            if "noise" in self.args.WT:
                episilon = episilon - z
        
        # Compute diffusion weighting
        if "VDS" in self.args.WT:
            wt = std**2
        elif "stable" in self.args.WT:
            wt = 1.0
        elif "score" in self.args.WT:
            wt = alpha_t / std
        else:
            assert False, "Unknown weighting type"
            
        # Compute cost Q-function guidance
        detach_a = a.detach().requires_grad_(True)
        
        # Handle different cost critic modes
        if self.args.qc_mode == "twin":
            qs = self.qc[0].q0_target.both(detach_a, s)
            q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0  # Average twin values
        elif self.args.qc_mode == "ensemble":
            qs = self.qc[0].q0_target.all(detach_a, s)
            q = sum(t.squeeze() for t in qs) / float(len(qs))  # Average ensemble
            
        self.policy.q = torch.mean(q)  # Store for monitoring
        
        # Compute gradients of cost Q-function (guidance to avoid costly actions)
        guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        
        # Optional gradient normalization
        if self.args.regq:
            guidance_norm = torch.mean(guidance**2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        # Loss: score matching + cost penalty (positive guidance increases cost)
        loss = (episilon * a).sum(-1) * wt + (guidance * a).sum(-1) * self.args.beta

        # Optimize policy
        loss = loss.mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()
        self.diffusion_behavior.train()

        return self.policy.q

    def update_policy_cost_lagrangian(self, data):
        """
        Update policy using Lagrangian-based safe optimization.
        
        This method implements constrained optimization where the policy maximizes
        reward while satisfying cost constraints. It uses Lagrangian multipliers
        that are adapted via PID control based on constraint violations.
        
        The optimization objective is:
        maximize: Q(s,a) - λ * Qc(s,a)
        where λ is the Lagrangian multiplier controlling constraint tightness.
        
        Args:
            data (dict): Batch of data containing states
        """
        s = data["s"]
        self.diffusion_behavior.eval()
        
        # Generate actions and setup diffusion process
        a = self.policy.act(s)
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        
        # Get diffusion model predictions
        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()
            if "noise" in self.args.WT:
                episilon = episilon - z
                
        # Compute diffusion weighting
        if "VDS" in self.args.WT:
            wt = std**2
        elif "stable" in self.args.WT:
            wt = 1.0
        elif "score" in self.args.WT:
            wt = alpha_t / std
        else:
            assert False, "Unknown weighting type"

        # Compute value and cost predictions
        detach_a = a.detach().requires_grad_(True)
        
        # Reward Q-function prediction
        qs = self.q[0].q0_target.both(detach_a, s)
        q_val = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        
        # Cost Q-function prediction (handle different modes)
        if self.args.qc_mode == "twin":
            qc_s = self.qc[0].q0_target.both(detach_a, s)
            qc_val = sum(t.squeeze() for t in qc_s) / float(len(qc_s))
        elif self.args.qc_mode == "ensemble":
            qc_s = self.qc[0].q0_target.all(detach_a, s)
            qc_val = sum(t.squeeze() for t in qc_s) / float(len(qc_s))
            
        # Compute Lagrangian multiplier
        if self.args.pid:
            # Use PID controller to adapt multiplier based on constraint violations
            print("using PID controller")
            with torch.no_grad():
                multiplier = self.controller.control(qc_val).detach()
        else:
            # Fixed multiplier
            multiplier = 0.1
            
        # Compute combined guidance: maximize reward - λ * cost
        # This creates gradients that push towards high reward, low cost actions
        guidance = torch.autograd.grad(
            torch.sum(q_val) - torch.sum(qc_val) * multiplier, detach_a
        )[0].detach()
        
        # Final loss: score matching + Lagrangian guidance
        loss = (episilon * a).sum(-1) * wt - (guidance * a).sum(-1) * self.args.beta
        
        # Optimize policy
        loss = loss.mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()
        self.diffusion_behavior.train()

    def update_critic(self, data, TD=False):
        """
        Update both reward and cost critics.
        
        Args:
            data (dict): Batch of offline data with states, actions, rewards, costs
            TD (bool): If True, use temporal difference updates; otherwise use IQL updates
        """
        if not TD:
            # Use IQL (Implicit Q-Learning) updates
            self.q[0].update_q0(data, policy=self.policy, object="reward")
            self.qc[0].update_q0(data, policy=self.policy, object="cost")
        else:
            # Use standard TD updates
            self.q[0].update_critic_td(data, policy=self.policy, object="reward")
            self.qc[0].update_critic_td(data, policy=self.policy, object="cost")

    def update_target_critic(self):
        """
        Soft update of target networks for both reward and cost critics.
        Uses exponential moving average with tau=0.005 for stability.
        """
        for q, qc in zip(self.q, self.qc):
            update_target(q.q0, q.q0_target, 0.005)
            update_target(qc.q0, qc.q0_target, 0.005)

    def get_qc_UCB(self, data):
        """
        Compute Upper Confidence Bound (UCB) for cost Q-function.
        
        This provides an optimistic estimate of costs for uncertainty-aware
        safe exploration by adding k * standard_deviation to the mean estimate.
        
        Args:
            data (dict): Batch containing states
            
        Returns:
            UCB estimate of cost Q-values
        """
        s = data["s"]
        with torch.no_grad():
            a = self.policy.act(s).detach()
            # Compute mean + k * std across ensemble members
            qc_UCB = torch.mean(
                torch.stack(self.qc[0].q0.all(a, s))
            ) + self.args.k * torch.std(torch.stack(self.qc[0].q0.all(a, s)))
        return qc_UCB


class SRPO_Behavior(nn.Module):
    """
    SRPO Behavior Model for learning diffusion-based action generation.
    
    This class implements the behavioral component of SRPO that learns to generate
    actions using a diffusion model. It's trained on offline data to capture the
    data distribution and can generate diverse, realistic actions.
    
    Args:
        input_dim (int): Dimension of state-action space
        output_dim (int): Dimension of action space
        marginal_prob_std (callable): Diffusion noise schedule function
        args: Configuration arguments
    """
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        
        # Initialize diffusion model for behavior learning
        self.diffusion_behavior = ScoreNet_IDQL(
            input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args
        )
        self.diffusion_optimizer = torch.optim.AdamW(
            self.diffusion_behavior.parameters(), lr=3e-4
        )
        
        # Store configuration
        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.inference_step = args.inference_step  # Number of denoising steps

    def update_behavior(self, data):
        """
        Update the diffusion behavior model using offline data.
        
        Trains the model to predict noise that was added to actions at random
        timesteps. This enables the model to learn the reverse diffusion process.
        
        Args:
            data (dict): Batch containing actions (a) and states (s)
        """
        self.step += 1
        all_a = data["a"]  # Ground truth actions
        all_s = data["s"]  # Corresponding states

        # Set model to training mode
        self.diffusion_behavior.train()

        # Sample random timesteps (avoid t=0 and t=1 for numerical stability)
        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1.0 - 1e-3) + 1e-3
        
        # Sample noise and create noisy actions
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        
        # Predict the noise that was added
        episilon = self.diffusion_behavior(perturbed_x, random_t, all_s)
        
        # Loss: MSE between predicted and actual noise
        loss = torch.mean(torch.sum((episilon - z) ** 2, dim=(1,)))
        self.loss = loss

        # Optimize the diffusion model
        self.diffusion_optimizer.zero_grad()
        loss.backward()
        self.diffusion_optimizer.step()

    def sample_action(self, state):
        """
        Generate an action using the trained diffusion model via reverse diffusion.
        
        This method implements the reverse diffusion process, starting from Gaussian
        noise and iteratively denoising to produce a realistic action. Each step uses
        the trained model to predict and remove noise.
        
        Args:
            state (torch.Tensor): Current state
            
        Returns:
            torch.Tensor: Generated action
        """
        self.diffusion_behavior.eval()

        # Start from pure Gaussian noise
        z = torch.randn(self.output_dim, device=self.args.device)
        z = z.unsqueeze(0)  # Add batch dimension

        # Reverse diffusion process: iteratively denoise
        for t in reversed(
            range(1, self.inference_step + 1)
        ):  # Go from T to 1 (avoid t=0)
            # Current timestep as fraction of total steps
            random_t = torch.tensor([t / self.inference_step], device=self.args.device)
            
            # Predict noise at current timestep
            with torch.no_grad():
                epsilon = self.diffusion_behavior(z, random_t, state)
                
            # Get diffusion schedule parameters
            alpha_t, std_t = self.marginal_prob_std(random_t)

            # Check for numerical issues that could cause instability
            if (
                torch.isnan(epsilon).any()
                or torch.isnan(alpha_t).any()
                or torch.isnan(std_t).any()
            ):
                print(f"NaN detected at step {t}")
                print(f"epsilon: {epsilon}")
                print(f"alpha_t: {alpha_t}")
                print(f"std_t: {std_t}")
                break

            # Clamp values for numerical stability
            std_t = torch.clamp(std_t, min=1e-6)
            alpha_t = torch.clamp(alpha_t, min=1e-6)

            # Reverse diffusion step: remove predicted noise
            z = (z - epsilon * std_t[:, None]) / alpha_t[:, None]

            # Additional check for NaN in the denoised result
            if torch.isnan(z).any():
                print(f"NaN detected in z at step {t}")
                print(f"z: {z}")
                break

        return z  # Return the final denoised action

    def act(self, state):
        """
        Action selection interface - calls sample_action.
        
        Args:
            state (torch.Tensor): Current state
            
        Returns:
            torch.Tensor: Action to take
        """
        return self.sample_action(state)


class SRPO_IQL(nn.Module):
    """
    SRPO with Implicit Q-Learning (IQL) baseline implementation.
    
    This class provides a comparison baseline that uses standard IQL for policy
    learning instead of diffusion-based optimization. It includes both reward
    and cost critics for safe reinforcement learning.
    
    Args:
        input_dim (int): Dimension of state-action space
        output_dim (int): Dimension of action space  
        args: Configuration arguments
    """
    def __init__(self, input_dim, output_dim, args=None):
        super().__init__()
        
        # Initialize deterministic policy
        self.policy = Dirac_Policy(output_dim, input_dim - output_dim).to("cuda")
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        # Cosine annealing scheduler for policy learning rate
        self.policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=1500000, eta_min=0.0
        )

        self.args = args
        self.output_dim = output_dim
        self.step = 0
        
        # Initialize reward critic
        self.q = []
        self.q.append(
            IQL_Critic(
                adim=output_dim,
                sdim=input_dim - output_dim,
                args=args,
                mode=args.q_mode,
            )
        )
        
        # Initialize cost critic for safety
        self.qc = []
        self.qc.append(
            IQL_Critic(
                adim=output_dim,
                sdim=input_dim - output_dim,
                args=args,
                mode=args.qc_mode,
            )
        )

    def update_iql(self, data):
        """
        Update IQL components: Q-functions and policy.
        
        This method updates both reward and cost Q-functions using IQL, then
        updates the policy using advantage-weighted regression (AWR).
        
        Args:
            data (dict): Batch of offline data
        """
        a = data["a"]
        s = data["s"]
        
        # Update reward Q-function using IQL
        self.q[0].update_q0(data)
        # Update cost Q-function using IQL
        self.qc[0].update_q0(data, object="cost")

        # Note: Alternative TD-based update methods available:
        # self.qc[0].update_critic_td(data, policy=self.policy, object='cost')
        # self.qc[0].update_q0(data, object='cost')
        
        # Policy update using advantage-weighted regression
        with torch.no_grad():
            # Get target Q-values and state values
            target_q = self.q[0].q0_target(a, s).detach()
            v = self.q[0].vf(s).detach()
            
        # Compute advantages
        adv = target_q - v
        temp = 3.0  # Temperature parameter for advantage weighting
        # Exponential weighting of advantages (clamped for stability)
        exp_adv = torch.exp(temp * adv.detach()).clamp(max=100.0)

        # Policy outputs and behavior cloning loss
        policy_out = self.policy(s)
        bc_losses = torch.sum((policy_out - a) ** 2, dim=1)
        
        # Advantage-weighted policy loss
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)
        
        # Optimize policy
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()
        
        self.policy_loss = policy_loss  # Store for monitoring


def update_target(new, target, tau):
    """
    Soft update of target network parameters.
    
    Performs exponential moving average update: target = tau * new + (1-tau) * target
    This provides more stable training compared to hard updates.
    
    Args:
        new (nn.Module): Source network with updated parameters
        target (nn.Module): Target network to be updated
        tau (float): Update rate (small values for slow updates)
    """
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def asymmetric_l2_loss(u, tau):
    """
    Asymmetric L2 loss for quantile regression in IQL.
    
    This loss function is used for value function learning in IQL, providing
    asymmetric penalties that help with distributional value estimation.
    
    Args:
        u (torch.Tensor): Prediction errors (predicted - target)
        tau (float): Quantile level (typically 0.7 for IQL)
        
    Returns:
        torch.Tensor: Asymmetric loss value
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL_Critic(nn.Module):
    """
    Implicit Q-Learning (IQL) Critic implementation.
    
    This class implements the critic networks for IQL, supporting both twin Q-learning
    (two Q-networks) and ensemble methods (multiple Q-networks). It includes both
    Q-function and value function components.
    
    The IQL approach avoids distributional shift by using the value function
    to provide a stable target for Q-learning updates.
    
    Args:
        adim (int): Action dimension
        sdim (int): State dimension
        args: Configuration arguments
        mode (str): "twin" for twin Q-learning, "ensemble" for ensemble methods
    """
    def __init__(self, adim, sdim, args, mode="twin") -> None:
        super().__init__()
        self.mode = mode
        
        # Initialize Q-network(s) based on mode
        if mode == "twin":
            # Twin Q-learning: two Q-networks for reduced overestimation
            self.q0 = TwinQ(adim, sdim, layers=args.q_layer).to(args.device)
        else:
            # Ensemble Q-learning: multiple Q-networks for uncertainty estimation
            print("import ensemble q-learning for qc")
            self.q0 = EnsembleQ(adim, sdim, layers=args.q_layer).to(args.device)

        # Create target Q-network (frozen copy for stable targets)
        self.q0_target = copy.deepcopy(self.q0).to(args.device)

        # Value function for IQL (predicts state values)
        self.vf = ValueFunction(sdim).to("cuda")
        
        # Optimizers for Q-function and value function
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=args.q_lr)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=args.v_lr)
        
        # IQL hyperparameters
        self.discount = 0.99  # Discount factor for future rewards
        self.args = args
        self.tau = 0.7  # Quantile level for asymmetric loss
        print("printing IQL critic tau value", self.tau)

    def update_q0(self, data, policy=None, object="reward"):
        """
        Update Q-function using IQL approach.
        
        IQL updates the Q-function using the current value function as a baseline,
        avoiding distributional shift issues in offline RL. The method alternates
        between updating the value function and Q-function.
        
        Args:
            data (dict): Batch containing states (s), actions (a), rewards/costs (r/c),
                        next states (s_), and done flags (d)
            policy (nn.Module, optional): Policy for action selection (if None, use data actions)
            object (str): "reward" or "cost" to specify which signal to optimize
        """
        s = data["s"]
        
        # Use policy actions if provided, otherwise use data actions
        if policy is None:
            a = data["a"]
        else:
            with torch.no_grad():
                a = policy.act(s).detach()

        # Select reward or cost signal
        if object == "reward":
            r = data["r"]
        elif object == "cost":
            r = data["c"]
        else:
            raise ValueError("wrong objective, optimize reward or cost")
            
        s_ = data["s_"]  # Next states
        d = data["d"]    # Done flags
        
        # Prepare targets for value function update
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()  # Current Q-value
            next_v = self.vf(s_).detach()             # Next state value
            
        # Update value function using quantile regression
        v = self.vf(s)  # Current state value
        adv = target_q - v  # Advantage (Q - V)
        # Use asymmetric loss for robust value estimation
        v_loss = asymmetric_l2_loss(adv, self.tau)
        
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q-function using Bellman equation with value function target
        targets = r + (1.0 - d.float()) * self.discount * next_v.detach()
        
        # Get Q-values based on network mode
        mode = self.mode
        if mode == "twin":
            qs = self.q0.both(a, s)  # Both Q-networks
        else:
            qs = self.q0.all(a, s)   # All ensemble members

        # Store diagnostics
        self.v = v.mean()
        
        # Compute Q-loss as MSE between predictions and targets
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        
        # Store losses and values for monitoring
        self.v_loss = v_loss
        self.q_loss = q_loss
        self.q = target_q.mean()
        self.v = next_v.mean()
        
        # Soft update of target Q-network
        update_target(self.q0, self.q0_target, 0.005)

    def update_critic_td(self, data, policy, object="reward"):
        """
        Update the critic using standard Temporal Difference (TD) learning.
        
        This method provides an alternative to IQL updates using traditional
        TD learning with policy-generated next actions. It combines value function
        updates (using quantile regression) with Q-function updates (using TD targets).
        
        Args:
            data (dict): Offline data containing states, actions, rewards/costs, 
                        next states, and done flags
            policy (nn.Module): Policy for generating next actions
            object (str): "reward" or "cost" to specify optimization target
        """
        # Extract data components
        s = data["s"]   # Current states
        a = data["a"]   # Current actions
        
        # Select reward or cost signal
        if object == "reward":
            r = data["r"]
        elif object == "cost":
            r = data["c"]
        else:
            raise ValueError("Not the right mode; Optimize reward or cost")
            
        s_ = data["s_"]  # Next states
        d = data["d"]    # Done flags

        # Update value function (same as in IQL)
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            
        v = self.vf(s)
        adv = target_q - v
        # Use quantile regression for robust value estimation
        v_loss = asymmetric_l2_loss(adv, self.tau)
        
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q-function using TD learning
        mode = self.mode
        
        # Get current Q-values
        if mode == "twin":
            qs_current = self.q0.both(a, s)
        else:
            qs_current = self.q0.all(a, s)
            
        r = r.view(-1, 1)  # Reshape rewards for broadcasting
        
        # Compute next Q-values using current policy (TD target)
        with torch.no_grad():
            a_ = policy.act(s_)  # Next actions from current policy
            
            if mode == "twin":
                qs_next = self.q0_target.both(a_, s_)
            else:
                qs_next = self.q0_target.all(a_, s_)

        # Compute TD targets: r + γ * Q(s', π(s'))
        td_targets = [
            r + (1.0 - d.float()) * self.discount * q_next for q_next in qs_next
        ]

        # Q-function loss: MSE between current Q and TD targets
        q_loss = sum(
            torch.nn.functional.mse_loss(q, target)
            for q, target in zip(qs_current, td_targets)
        ) / len(qs_current)
        
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Store loss for monitoring
        self.q_loss = q_loss
        
        # Soft update of target Q-network
        update_target(self.q0, self.q0_target, 0.005)
