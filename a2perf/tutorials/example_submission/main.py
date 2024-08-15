import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from a2perf.domains import quadruped_locomotion  # noqa: F401

# env = gym.make('QuadrupedLocomotion-v0')
# Parallel environments
vec_env = make_vec_env("QuadrupedLocomotion-v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=250)
model.save("ppo_cartpole")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
env = gym.make("QuadrupedLocomotion-v0", enable_rendering=True)
env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
