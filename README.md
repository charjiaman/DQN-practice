# 🧠 DQN LunarLander with Video Recording

This project demonstrates how to train and evaluate a Deep Q-Network (DQN) agent using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/). It uses the classic `LunarLander-v3` environment from [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

## 🛠 Project Features

- ✅ Train a DQN agent from scratch using `train.py`
- ✅ Test and **record a video** of the trained agent using `test.py`
- ✅ Saves:
  - The **trained model** (`dqn_lunar.zip`)
  - The **gameplay video** in `./videos/`

> ⚡ **If you feel training takes too long**, you can **run `test.py` directly** to view the results using the saved model and see the agent’s behavior immediately.

---

## 🎯 Environment Overview

- **Environment**: `LunarLander-v3`
- **Reward function details**:  
  https://gymnasium.farama.org/environments/box2d/lunar_lander/

---

## 💻 Setup Instructions (Windows + Anaconda)

1. **Open Anaconda Prompt** from the Start Menu  
2. **Right-click** in the terminal and **paste** the following instructions:

```bash
# Step 1: Create the conda environment from YAML
conda env create -f rl-dqn.yml

# Step 2: Activate the environment
conda activate rl-dqn
