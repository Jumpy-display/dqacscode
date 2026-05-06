import os
import time
import random
import numpy as np
import gymnasium as gym
import metaworld
import torch
import csv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import sys
from pathlib import Path
cwd = Path(__file__).parent.resolve()
sys.path.append(cwd)

from finalAgent3_metaworld import make_metaworld_env, HybridContinuousAgent, RunningMeanStd, seed_everything
from finalAgent3_metaworld import BATCH_SIZE, NUM_ENVS, HIDDEN_DIM



# =============================================================================
# Setup
# =============================================================================
# PPO: use cpu is more efficient for MLP
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


env_name = "box-close-v3"
SEED = 1
TOTAL_TIMESTEPS = 20_000_000
print('ENV:', env_name)


# =============================================================================
# Vectorized environment
# =============================================================================
# env = gym.vector.SyncVectorEnv(
# 	[make_metaworld_env(env_name, SEED, i) for i in range(NUM_ENVS)]
# )
# env = make_vec_env(
#     lambda: make_metaworld_env(env_name, SEED, 0), 
#     n_envs=NUM_ENVS
# )
env_fns = [make_metaworld_env(env_name, SEED, idx) for idx in range(NUM_ENVS)]
# env = SubprocVecEnv(env_fns)
env = DummyVecEnv(env_fns)
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
)
# =============================================================================
# Logging callback (matches your style)
# =============================================================================
class LogCallback(BaseCallback):
	def __init__(self, log_path="ppo_log", window=50, save_freq=1_000_000, verbose=0):
		super().__init__(verbose)
		self.start_time = time.time()
		self.window = window
		self.episode_rewards = []
		self.episode_success = []
		self.log_path = log_path
		self.csv_path = os.path.join(log_path, "training_log.csv")
		self.save_freq = save_freq

		with open(self.csv_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["steps", "avg_reward", "success_rate", "sps"])

	def _on_training_start(self):
		n_envs = self.training_env.num_envs
		self.current_returns = np.zeros(n_envs)
		self.current_success = np.zeros(n_envs)

	def _on_step(self) -> bool:
		rewards = self.locals.get("rewards", [])
		dones = self.locals.get("dones", [])
		infos = self.locals.get("infos", [])

		self.current_returns += rewards

		for i, info in enumerate(infos):
			self.current_success[i] = max(
				self.current_success[i],
				float(info.get("success", 0.0))
			)

			if dones[i]:
				self.episode_rewards.append(self.current_returns[i])
				self.episode_success.append(self.current_success[i])

				self.current_returns[i] = 0.0
				self.current_success[i] = 0.0

		# print(self.num_timesteps)
		if self.n_calls % max(1, 10_000 // self.training_env.num_envs) == 0:
			print("Global steps:", self.num_timesteps)

		if self.n_calls % (100_000 // self.training_env.num_envs) == 0:
			recent_rewards = self.episode_rewards[-self.window:]
			avg_reward = np.mean(recent_rewards)

			recent_success = self.episode_success[-self.window:]
			success_rate = np.mean(recent_success)

			sps = int(self.num_timesteps / (time.time() - self.start_time))

			print(
				f"[PPO] steps={self.num_timesteps} | "
				f"reward={avg_reward:.2f} | "
				f"success={success_rate*100:.1f}% | "
				f"SPS={sps}"
			)

			with open(self.csv_path, "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerow([
					self.num_timesteps,
					avg_reward,
					success_rate,
					sps
				])

		if (
			self.num_timesteps > 0
			and self.num_timesteps % self.save_freq < self.training_env.num_envs
		):
			ckpt_path = f"{self.log_path}/ppo_ckpt_{self.num_timesteps}_steps.zip"
			print(f"[Checkpoint] Saving model at {self.num_timesteps} steps -> {ckpt_path}")
			self.model.save(ckpt_path)

		return True

# =============================================================================
# PPO model (official baseline)
# =============================================================================
model = PPO(
	"MlpPolicy",
	env,
	learning_rate=3e-4,
	n_steps=2048,  # PPO uses n_steps instead of buffer_size
	batch_size=BATCH_SIZE,
	gamma=0.99,
	gae_lambda=0.95,
	ent_coef=0.0,
	vf_coef=0.5,
	max_grad_norm=0.5,
	verbose=0,
	device=DEVICE,
	policy_kwargs=dict(net_arch=[HIDDEN_DIM, HIDDEN_DIM])
)


# =============================================================================
# Train
# =============================================================================
log_path = f"PPO_runs/metaworld_{env_name}_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_path, exist_ok=True)

callback = LogCallback(log_path=log_path)

model.learn(
	total_timesteps=TOTAL_TIMESTEPS,
	callback=callback
)


# =============================================================================
# Save model
# =============================================================================
os.makedirs("baselines", exist_ok=True)
model.save(f"baselines/ppo_{env_name}")

print("Training complete.")
env.close()