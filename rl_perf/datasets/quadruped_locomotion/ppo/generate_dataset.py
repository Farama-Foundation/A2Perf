import glob
import pickle
from absl import app, flags
import numpy as np
import minari
from minari import DataCollectorV0, StepDataCallback
import gymnasium as gym
from rl_perf.domains import quadruped_locomotion
from rl_perf.domains.quadruped_locomotion.motion_imitation.learning import ppo_imitation
import os

FLAGS = flags.FLAGS

# Define command line flags.
flags.DEFINE_string('glob_pattern', '/mnt/gcs/a2perf/quadruped_locomotion/dog_pace/ppo/0006/**/*_steps.zip',
                    'Glob pattern to match policy files')
flags.DEFINE_integer('num_samples', 100, 'Number of samples to generate for each policy')

# Ensure that the flags are mandatory.
flags.mark_flag_as_required('glob_pattern')


def load_model(policy_path, env):
    model = ppo_imitation.PPOImitation.load(policy_path, env=env)
    return model


def infer_once(model, observation):
    action, _states = model.predict(observation)
    return action


def preprocess_observation(observation):
    return observation


def main(argv):
    del argv  # Unused.
    env = gym.make('QuadrupedLocomotion-v0')

    # Get all saved models
    policy_files = glob.glob(FLAGS.glob_pattern, recursive=True)

    # Group all saved models by their path
    policy_files = [os.path.dirname(x) for x in policy_files]
    policy_files = set(policy_files)

    transitions = []

    for policy_path in policy_files:

        # get the latest policy by finding max number in the policy path
        max_policy = max([int(x.split('_')[2]) for x in os.listdir(policy_path)])

        model = load_model(max_policy, env)
        for _ in range(FLAGS.num_samples):
            observation = env.reset()  # Reset the environment to get the initial observation.
            observation = preprocess_observation(observation)
            action = infer_once(model, observation)
            next_observation, reward, terminated, truncated, info = env.step(action)  # Step the environment with the action.
            done = truncated or terminated
            transition = (observation, action, reward, next_observation, done)
            transitions.append(transition)

    # Here you can save the transitions to a file or a database.
    # For example:
    with open('transitions.pkl', 'wb') as f:
        pickle.dump(transitions, f)


if __name__ == '__main__':
    app.run(main)
