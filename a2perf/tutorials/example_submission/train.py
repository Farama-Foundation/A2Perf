# 1) Import the necessary packages.
# For this tutorial, we will use the `quadruped_locomotion` environment.
# import time
from typing import Callable

# import gymnasium to create the environment
import gymnasium as gym

# import snntorch as snn
import torch
import torch.nn as nn

# import the abseil app to run the experiment
# from absl import app
from gymnasium import spaces

# import packages needed for your training
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

# import the relevant A2Perf domain
# import a2perf.domains.quadruped_locomotion
import a2perf.domains.web_navigation.gwob.CoDE  # noqa: F401
from a2perf.domains.web_navigation.gwob.CoDE import environment  # noqa: F401

# import gin


# from sb3_contrib import RecurrentPPO
# from snntorch import surrogate


# print all registered gym environments
print("Creating environment")
env = gym.make(
    "WebNavigation-v0", num_websites=2, difficulty=2, raw_state=True, step_limit=7
)


env_low = env.env.unwrapped
miniwobstate = env.reset()[0]
print(miniwobstate)
print("Utterance")
print(miniwobstate.utterance)
print("Phrase")
print(miniwobstate.phrase)
print("Tokens")
print(miniwobstate.tokens)
print("Fields")
print(miniwobstate.fields)
print("Dom")
print(miniwobstate.dom)
print("Dom elements")
print(miniwobstate.dom_elements)


class CustomNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(CustomNet, self).__init__()

        self.latent_dim_pi = action_shape.n
        self.latent_dim_vf = 1
        self.actor = nn.Sequential(
            nn.Linear(state_shape.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape.n),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_shape.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features)

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNet(self.observation_space, self.action_space)


# 2) Next, we define our training function.
# This function will be called by the abseil app.
def train():
    # ac = CustomActorCriticPolicy(env.observation_space.shape, env.action_space)
    """Include your training algorithm here."""
    # Create the environment
    vec_env = make_vec_env("CartPole-v1", n_envs=8)

    # Create the agent
    model = PPO(CustomActorCriticPolicy, vec_env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=25e3)
    # Save the agent
    model.save("ppo_cartpole")

    del model  # remove to demonstrate saving and loading


# 3) Optionally, we define the main function.
# This function will be called when the script is run directly.
def main(_):
    # The main function where the training process is initiated.
    train()


if __name__ == "__main__":
    # Run the main function using the abseil app.
    # This allows us to pass command line arguments to the script.
    #   app.run(main)
    pass
