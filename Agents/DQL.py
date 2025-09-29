import copy

import torch
import torch.nn as nn

from Agents.model import *
from Agents.SRPO import IQL_Critic

class DQL(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None, scale = 0.01):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        self.policy = Gaussian_Simple_Policy(output_dim, input_dim-output_dim, layer=args.policy_layer, scale= scale).to("cuda")
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.qc = []
        self.q.append(IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
        self.qc.append(IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
    
    
    def update_policy(self, data):
        s = data['s']
        self.diffusion_behavior.eval()
        a = self.policy.act(s)
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()
            if "noise" in self.args.WT:
                episilon = episilon - z
        if "VDS" in self.args.WT:
            wt = std ** 2
        elif "stable" in self.args.WT:
            wt = 1.0
        elif "score" in self.args.WT:
            wt = alpha_t / std
        else:
            assert False
        detach_a = a.detach().requires_grad_(True)
        qs = self.q[0].q0_target.both(detach_a , s)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.policy.q = torch.mean(q)
        guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm
        loss = (episilon * a).sum(-1) * wt  - (guidance * a).sum(-1) * self.args.beta
        loss = loss.mean() 
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()
        self.diffusion_behavior.train()
        
    
    def update_critic(self, data):
        pass 



class DQGL(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        self.policy = Gaussian_Policy(output_dim, input_dim-output_dim, layer=args.policy_layer).to("cuda")
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.qc = []
        self.q.append(IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
        self.qc.append(IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
    
    
    def update_policy(self, data, entropy=False):
        s = data['s']
        self.diffusion_behavior.eval()
        a = self.policy.act(s,deterministic=False)
        _, a_log_std, _ = self.policy(s)
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()
            if "noise" in self.args.WT:
                episilon = episilon - z
        if "VDS" in self.args.WT:
            wt = std ** 2
        elif "stable" in self.args.WT:
            wt = 1.0
        elif "score" in self.args.WT:
            wt = alpha_t / std
        else:
            assert False
        detach_a = a.detach().requires_grad_(True)
        qs = self.q[0].q0_target.both(detach_a , s)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.policy.q = torch.mean(q)
        # TODO be aware that there is a small std gap term here, this seem won't affect final performance though
        # guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach() * std[..., None]
        guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm
        if entropy:
            loss = (episilon * a).sum(-1) * wt   - (guidance * a).sum(-1) * self.args.beta
        else:
            loss = (episilon * a).sum(-1) * wt   - (guidance * a).sum(-1) * self.args.beta + a_log_std.sum(-1)*wt
        loss = loss.mean() 
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()
        self.diffusion_behavior.train()
    
    
    def update_critic(self, data):
        a = data['a']
        s = data['s']
        self.q[0].update_q0(data)
        
        data_cost = copy.deepcopy(data)
        data_cost['r'] = copy.deepcopy(data['c'])
        self.qc[0].update_q0(data_cost)
        
        # evaluate iql policy part, can be deleted
        with torch.no_grad():
            target_q = self.q[0].q0_target(a, s).detach()
            v = self.q[0].vf(s).detach()
        adv = target_q - v
        temp = 10.0 if "maze" in self.args.env else 3.0
        exp_adv = torch.exp(temp * adv.detach()).clamp(max=100.0)

        policy_out = self.deter_policy(s)
        bc_losses = torch.sum((policy_out - a)**2, dim=1)
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)
        self.deter_policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.deter_policy_optimizer.step()
        self.deter_policy_lr_scheduler.step()
        self.policy_loss = policy_loss 
    

def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


