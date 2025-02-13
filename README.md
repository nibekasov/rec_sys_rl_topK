# Reinforcement Learning for Recommendation Systems

## Overview
This project implements a reinforcement learning (RL) approach for recommendation systems. The goal is to improve recommendations by leveraging off-policy corrections and REINFORCE with Top-K corrections.
It is insoired by the paper https://arxiv.org/pdf/1812.02353 and use from repo ()

## Features
- **Actor-Critic Model**: Implements a policy network (actor) and a value network (critic) using PyTorch.
- **Off-Policy Corrections**: Implements an off-policy correction mechanism to improve learning stability.
- **Logging & Monitoring**: Uses `wandb` for experiment tracking and logging.
- **Top-K Reinforcement Learning**: Implements the REINFORCE algorithm with Top-K correction for better sample efficiency.
- **Data Handling**: Supports efficient batch processing of recommendation datasets.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, Pandas
- Weights & Biases (wandb)
- torch_optimizer

### Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/reinforce-recsys.git
cd reinforce-recsys
pip install -r requirements.txt
```

## Usage
### Training
Run the main script to start training:

```bash
python main.py --epochs 20 --batch_size 32 --lr 0.001
```

### Monitoring with W&B
To track experiments, log in to wandb:
```bash
wandb login
```
Then, start tracking runs automatically within the script.

## Project Structure
```
reinforce-recsys/
├── data/                   # Dataset files
├── models/                 # Trained models
├── utils/                  # Utility scripts
├── main.py                 # Main script
├── model.py                # Neural network definitions
├── reinforce.py            # Reinforcement learning logic
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```
## Logs of training process

https://wandb.ai/nibekasov-itmo-university/reinforce-topK/runs/2jmlmsqo?nw=nwusernibekasov

## ToDO: make better implementation in RecSys
## Refactor code
## ToDo: Refactor usage of unnesesary libs

## Contribution
Feel free to open issues and pull requests if you want to improve this repository!

## License
MIT License. See `LICENSE` for details.

