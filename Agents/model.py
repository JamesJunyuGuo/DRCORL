import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Gaussian random Fourier features for encoding a scalar (time) input.

    This layer maps an input ``t`` (e.g. diffusion time or noise level) to a
    higher dimensional periodic embedding via random projection followed by
    ``sin`` and ``cos``. The weights are sampled once and frozen (i.e. they are
    registered as a non-trainable parameter) which matches the common design in
    score-based generative models (see e.g. Song et al.).
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def mlp(dims, activation=nn.ReLU, output_activation=None):
    """Construct a simple feed-forward MLP.

    Parameters
    ----------
    dims : List[int]
        Layer dimensions including input and output. Example: ``[obs_dim, 256, 256, act_dim]``.
    activation : nn.Module class, optional
        Activation class (NOT instance) used for hidden layers. Default: ``nn.ReLU``.
    output_activation : nn.Module class, optional
        Optional activation applied only to the final layer (e.g. ``nn.Tanh`` for
        action squashing). If ``None`` no final non-linearity is appended.

    Returns
    -------
    torch.nn.Sequential
        The constructed network set to float32.
    """
    n_dims = len(dims)
    assert n_dims >= 2, "MLP requires at least two dims (input and output)"
    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class TwinQ(nn.Module):
    """Double Q-network architecture.

    Maintains two independent Q estimators (Q1, Q2) to mitigate positive bias
    in temporal difference targets (classic Double Q-learning idea). The
    forward pass returns the pointwise minimum which is widely used (e.g.
    TD3, SAC) for a conservative target.
    """

    def __init__(self, action_dim, state_dim, layers=2):
        super().__init__()
        dims = [state_dim + action_dim] + [256] * layers + [1]
        # dims = [state_dim + action_dim, 256, 256, 1] # TODO
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        """Return both Q-value predictions (Q1, Q2).

        Parameters
        ----------
        action : torch.Tensor
            Action tensor (batch, action_dim) possibly already concatenated.
        condition : torch.Tensor | None
            Conditional input (e.g. state). If provided will be concatenated
            with ``action`` along the last dimension.
        """
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        """Return the elementwise minimum of the two Q estimates."""
        return torch.min(*self.both(action, condition))


class EnsembleQ(nn.Module):
    """Ensemble of Q-functions.

    Provides a list of independently parameterized Q estimators. The forward
    method returns the minimum across the ensemble (a conservative estimate),
    while ``all`` returns the raw list which can be used for uncertainty-aware
    techniques (e.g., variance penalties, bootstrapping, CQL-style min, etc.).
    """

    def __init__(self, action_dim, state_dim, layers=2, ensemble_num=6):
        super().__init__()
        dims = [state_dim + action_dim] + [256] * layers + [1]

        self.qs = nn.ModuleList([mlp(dims) for _ in range(ensemble_num)])
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize each linear layer with Kaiming uniform and zero bias."""
        for q in self.qs:
            for layer in q:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=0, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def all(self, action, condition=None):
        """Compute list of Q-values across all ensemble members."""
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        q_vals = [q(as_) for q in self.qs]
        return q_vals

    def forward(self, action, condition=None):
        """Return elementwise minimum across ensemble for conservative value."""
        q_vals = self.all(action, condition)
        return torch.min(torch.stack(q_vals), dim=0).values


class ValueFunction(nn.Module):
    """State-value function V(s)."""

    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        """Return V(s)."""
        return self.v(state)


class Dirac_Policy(nn.Module):
    """Deterministic policy (Dirac distribution at mean action).

    Produces a squashed action via ``Tanh``. ``scale`` retained for interface
    parity with stochastic policies (not used directly here except stored).
    """

    def __init__(self, action_dim, state_dim, layer=2, scale=0.01):
        super().__init__()
        self.net = mlp(
            [state_dim] + [256] * layer + [action_dim], output_activation=nn.Tanh
        )
        self.scale = scale
        self.output_dim = action_dim

    def forward(self, state):
        """Return deterministic action (already squashed to [-1, 1])."""
        return self.net(state)

    def select_actions(self, state):
        """Alias for ``act`` used by some training loops expecting the name."""
        return self(state)

    def act(self, state):
        """Return action (deterministic)."""
        return self(state)


class Gaussian_Simple_Policy(nn.Module):
    """Stochastic Gaussian policy with fixed diagonal std.

    The log std is not learned; instead a constant ``scale`` is broadcast to the
    action dimension. ``forward`` returns (mean, log_std, std). Sampling uses
    reparameterization-like additive Gaussian noise (w/out explicit rsample).
    """

    def __init__(self, action_dim, state_dim, layer=2, scale=0.01):
        super().__init__()
        self.output_dim = action_dim
        self.action_mean = mlp(
            [state_dim] + [256] * layer + [action_dim], output_activation=nn.Tanh
        )
        self.scale = scale
        self.action_std = torch.full((1, action_dim), scale).to("cuda")
        self.action_log_std = torch.log(self.action_std)

    def forward(self, state):
        # Mean depends on state; std is a constant broadcast to match shape.
        action_mean = self.action_mean(state)
        action_std = torch.full_like(action_mean, self.scale)
        action_log_std = torch.log(action_std)
        return action_mean, action_log_std, action_std

    def select_actions(self, state):
        """Return deterministic action (the mean)."""
        return self.act(state, deterministic=True)

    def act(self, state, deterministic=False):
        """Sample (or return mean) action.

        Parameters
        ----------
        deterministic : bool, default False
            If True returns the mean action to support evaluation.
        """
        action_mean, _, action_std = self.forward(state)
        if deterministic:
            return action_mean
        else:
            return action_mean + self.action_std * torch.randn_like(action_mean)


class Gaussian_Policy(nn.Module):
    """Gaussian policy with learnable log std (state-independent).

    ``action_log_std`` is a parameter and expanded to match batch shape in the
    forward pass. This mirrors the standard SAC style actor (but w/out separate
    mean/std heads). ``scale`` retained for interface consistency.
    """

    def __init__(self, action_dim, state_dim, layer=2, scale=0.001):
        super().__init__()
        self.output_dim = action_dim
        self.action_mean = mlp(
            [state_dim] + [256] * layer + [action_dim], output_activation=nn.Tanh
        )
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.scale = scale

    def forward(self, state):
        # Expand log_std to batch shape & exponentiate to ensure positivity.
        action_mean = self.action_mean(state)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std

    def select_actions(self, state):
        """Return deterministic mean action."""
        return self.act(state, deterministic=True)

    def act(self, state, deterministic=False):
        """Sample action (mean + std * eps) or return mean if deterministic."""
        action_mean, _, action_std = self.forward(state)
        if deterministic:
            return action_mean
        else:
            return action_mean + action_std * torch.randn_like(action_mean)


class MLPResNetBlock(nn.Module):
    """Single residual block used by ``MLPResNet``.

    Structure: (optional dropout) -> (optional LayerNorm) -> Linear -> Act ->
    Linear -> Residual Add (with optional projection if shapes differ).
    """

    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)

        self.dropout = (
            nn.Dropout(dropout_rate)
            if dropout_rate is not None and dropout_rate > 0.0
            else None
        )

    def forward(self, x, training=False):  # training flag kept for API parity.
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x


class MLPResNet(nn.Module):
    """Residual MLP backbone.

    Composes ``num_blocks`` residual blocks after an input projection. A final
    linear layer maps to ``out_dim``. Activation function *after* each block is
    provided via ``activations`` (e.g. ``F.relu`` or a module instance).
    """
    def __init__(
        self,
        num_blocks,
        input_dim,
        out_dim,
        dropout_rate=None,
        use_layer_norm=False,
        hidden_dim=256,
        activations=F.relu,
    ):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim + 128, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(
                    self.hidden_dim,
                    self.activations,
                    self.dropout_rate,
                    self.use_layer_norm,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out_fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, training=False):
        x = self.fc(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.activations(x)
        x = self.out_fc(x)

        return x


class ScoreNet_IDQL(nn.Module):
    """Score / policy network with conditional time embedding.

    Combines a learnable conditional embedding (via Fourier features + MLP) and
    a residual MLP trunk. Designed for Implicit / Diffusion Q-Learning style
    methods where the network may predict a score, denoised action, or policy
    output conditioned on (state, time / noise level).
    """

    def __init__(
        self, input_dim, output_dim, marginal_prob_std, embed_dim=64, args=None
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim))
        self.device = args.device
        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.main = MLPResNet(
            args.actor_blocks,
            input_dim,
            output_dim,
            dropout_rate=0.1,
            use_layer_norm=True,
            hidden_dim=256,
            activations=nn.Mish(),
        )
        self.cond_model = mlp(
            [64, 128, 128], output_activation=None, activation=nn.Mish
        )

        # The swish activation function
        # self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t, condition):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (e.g., action or concatenated feature vector).
        t : torch.Tensor
            Time / noise level inputs (shape broadcastable to batch) to be
            embedded via random Fourier features.
        condition : torch.Tensor
            Additional conditioning (typically state / observation).
        """
        embed = self.cond_model(self.embed(t))
        all = torch.cat([x, condition, embed], dim=-1)
        h = self.main(all)
        return h


def build_mlp_network(sizes):
    """Utility for quickly building a tanh MLP.

    Hidden layers use ``nn.Tanh``. Final layer uses identity. Kaiming uniform
    initialization is applied to linear layers.
    """
    layers = list()
    for j in range(len(sizes) - 1):
        act = nn.Tanh if j < len(sizes) - 2 else nn.Identity
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        nn.init.kaiming_uniform_(affine_layer.weight, a=np.sqrt(5))
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)


class Value(nn.Module):
    """Simple state value network V(s)."""

    def __init__(self, obs_dim):
        super().__init__()
        self.critic = build_mlp_network([obs_dim, 64, 64, 1])

    def forward(self, obs):
        """Return squeezed value prediction (batch,)."""
        return torch.squeeze(self.critic(obs), -1)
