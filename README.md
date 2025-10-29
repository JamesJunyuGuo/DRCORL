
# Don't Trade Off Safety: Diffusion Regularization for Constrained Offline RL

**Junyu Guo, Zhi Zheng, Donghao Ying, Shangding Gu, Ming Jin, Costas Spanos, Javad Lavaei**  
*Neural Information Processing Systems (NeurIPS), 2025*

[![arXiv](https://img.shields.io/badge/arXiv-2502.12391-b31b1b.svg)](https://arxiv.org/abs/2502.12391) [![OpenReview](https://img.shields.io/badge/OpenReview-DRCORL-blue)](https://openreview.net/forum?id=eSIRst0WVy)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025%20Poster-4b8bbe.svg)](https://neurips.cc/virtual/2025/poster/116918)


---

## Overview

This is the official implementation of our NeurIPS 2025 paper. We propose a novel diffusion-based regularization approach for safe offline reinforcement learning that maintains safety constraints without sacrificing performance. Our method leverages diffusion models to learn safe behavioral priors from offline datasets, enabling effective constraint satisfaction in challenging safety-critical scenarios.

**Key Features:**
- 🎯 Diffusion-based policy regularization for constrained offline RL
- 🛡️ Safety constraint satisfaction without performance degradation
- 📊 State-of-the-art results on safety-critical benchmarks
- 🚀 Easy-to-use implementation with comprehensive documentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

A Safe Reinforcement Learning implementation using Diffusion Policies for constrained environments. This project combines the power of diffusion models with safe reinforcement learning to learn policies that respect safety constraints during both training and deployment.

## Overview

Safe Diffusion RL addresses the challenge of learning safe policies in offline reinforcement learning settings. The method uses diffusion models to represent stochastic policies while incorporating safety constraints through a dual critic architecture (reward and cost critics).

## Features

- 🛡️ **Safety-First**: Incorporates safety constraints directly into the learning process
- 🎯 **Diffusion Policies**: Uses score-based diffusion models for expressive policy representation
- 📊 **Dual Critic Architecture**: Separate reward and cost critics for balanced optimization
- 🏃 **Offline RL**: Learn from pre-collected datasets without environment interaction
- 🔧 **Modular Design**: Clean separation between agents, environments, and utilities

## Environment Setup

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable Python environment management.

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- [MuJoCo](https://github.com/deepmind/mujoco) physics simulator

### Quick Setup with uv

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Clone and setup the project**:
   ```bash
   git clone https://github.com/JamesJunyuGuo/DRCORL.git
   cd DRCORL
   
   # Create virtual environment and install dependencies
   uv venv --python 3.8
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

<!-- 3. **Install DSRL** (required dependency):
   ```bash
   uv pip install dsrl
   # OR install from source
   git clone https://github.com/liuzuxin/DSRL.git
   cd DSRL && uv pip install -e . && cd ..
   ``` -->

### Alternative Setup Methods

<details>
<summary>Using pip/conda</summary>

```bash
# Using pip
python -m venv DRCORL_env
source DRCORL_env/bin/activate  # On Windows: DRCORL_env\Scripts\activate
pip install -r requirements.txt

# Using conda
conda create -n DRCORL python=3.8
conda activate DRCORL
pip install -r requirements.txt
```
</details>

## Quick Start

### 1. Prepare Model Checkpoints

Download pretrained behavior and critic checkpoints and store them under `./Safe_model_factory/`, or train them yourself:

**Train Behavior Policy:**
```bash
TASK="OfflineCarCircle-v0"
seed=0
python safe_behavior.py --expid ${TASK}-baseline-seed${seed} \
                       --env $TASK \
                       --seed ${seed} \
                       --beta 0.2
```

**Train Safety Critic:**
```bash
python safe_critic.py --expid ${TASK}-baseline-seed${seed} \
                      --env $TASK \
                      --seed ${seed}
```

### 2. Train Safe Policy

```bash
TASK="OfflineCarCircle-v0"
seed=100
python safe_policy.py --expid ${TASK}-baseline-seed${seed} \
                     --env $TASK \
                     --seed ${seed} \
                     --actor_load_path ./Safe_model_factory/${TASK}-baseline-seed${seed}/behavior_ckpt200.pth \
                     --critic_load_path ./Safe_model_factory/${TASK}-baseline-seed${seed}/ \
                     --beta 0.2 \
                     --mode online \
                     --slack_bound 0.1 \
                     --cost_limit 1.0 \
                     --n_policy_epochs 1000
```

### 3. Run All Experiments

For convenience, run the complete pipeline:
```bash
bash Ant.sh
```

## Supported Environments

The project supports various safety-critical environments from the Safety Gymnasium suite. Available environments are listed in `utils/env_list.py`, including:

- `OfflineCarCircle-v0`: Navigation with circular obstacles
- `OfflineCarButton-v0`: Button pressing with safety constraints
- `OfflineAntVelocity-v0`: Ant locomotion with velocity constraints
- And many more...

## Project Structure

```
DRCORL/
├── Agents/                 # Core RL algorithms and models
│   ├── model.py           # Policy networks and score functions
│   ├── SRPO.py            # Safe RPO algorithm implementation
│   ├── DQL.py             # Diffusion Q-Learning
│   └── diffusion.py       # Diffusion model components
├── env/                   # Environment utilities
├── utils/                 # Helper functions and configurations
├── scripts/               # Training and evaluation scripts
├── safe_behavior.py       # Behavior policy training
├── safe_critic.py         # Safety critic training
├── safe_policy.py         # Main safe policy learning
└── requirements.txt       # Dependencies
```

## Algorithm Overview

Safe Diffusion RL consists of three main components:

1. **Behavior Policy Pretraining**: Learn a diffusion-based behavior policy from offline data
2. **Dual Critic Learning**: Train separate reward and cost critics to evaluate policy performance and safety
3. **Safe Policy Optimization**: Use the behavior policy and critics to learn a safe target policy

Unlike standard offline RL, the safety critic is continuously updated during policy learning to ensure the learned policy respects safety constraints.

## Development

### Code Quality

This project uses [black](https://github.com/psf/black) for code formatting and [pre-commit](https://pre-commit.com/) for automated quality checks:

```bash
# Install development dependencies
uv pip install black pre-commit

# Setup pre-commit hooks
pre-commit install

# Format code
black .
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests if applicable
4. Ensure code passes quality checks (`black . && pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex

``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DSRL](https://github.com/liuzuxin/DSRL) for the safety gymnasium environments
- [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) for safety-critical RL environments
- The PyTorch and JAX communities for excellent deep learning frameworks
