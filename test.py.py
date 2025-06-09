'''
Code reference:
    https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#id2
    Chatgpt
'''

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Step 1: Load environment (must be rgb_array for video)
env = DummyVecEnv([lambda: gym.make("LunarLander-v3", render_mode="rgb_array")])

# Step 2: Wrap with video recorder
video_folder = "./videos/"
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda x: x == 0,  # record only once
    video_length=2000,  # max steps allowed for video
    name_prefix="lunar-dqn"
)

# Step 3: Load trained model
model = DQN.load("dqn_lunar", env=env)

# Step 4: Run agent and record
obs = env.reset()
total_reward = 0
for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]  # DummyVecEnv wraps in array
    print(f"Step {step}: Reward={reward[0]:.2f}, Done={done[0]}")

    if done[0]:
        print("Episode finished.")
        break

# Step 5: Save video
env.close()
print(f"Total reward: {total_reward:.2f}")
print("Video saved to:", video_folder)

