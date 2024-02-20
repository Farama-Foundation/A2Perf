import collections
import json

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

NUM_POLICIES_PER_EXPERTISE_LEVEL = 50
EVAL_POINTS_PER_WINDOW = 5


def calculate_score_bounds_per_task(df, group_cols, reward_col):
  bounds_per_task = {}

  for group_keys, group in df.groupby(group_cols):
    mean_reward = group[reward_col].mean()
    std_reward = group[reward_col].std()

    bounds_per_task[group_keys] = {
        'novice': (-np.inf, mean_reward - (2 * std_reward)),
        'intermediate': (mean_reward - std_reward, mean_reward + std_reward),
        'expert': (mean_reward + (2 * std_reward), np.inf)
    }

  return bounds_per_task


def assign_expertise_level(row, bounds):
  """
  Assign expertise level to a row based on episode_reward and predefined bounds.
  """
  task_key = (row['domain'], row['task'])  # Create the key tuple for the task
  reward = row['episode_reward']

  # Check against the bounds and assign the level
  if reward <= bounds[task_key]['novice'][1]:
    return 'novice'
  elif bounds[task_key]['novice'][1] < reward <= \
      bounds[task_key]['intermediate'][1]:
    return 'intermediate'
  elif reward > bounds[task_key]['intermediate'][1]:
    return 'expert'


def plot_expertise_levels():
  for task, group in all_df.groupby('task'):
    plt.figure(figsize=(10, 6))  # Set the figure size for better visibility

    # Plot histograms for each expertise level on the same axes
    sns.histplot(data=group[group['expertise_level'] == 'novice'],
                 x='episode_reward', color='blue', label='Novice', kde=True,
                 stat="density", linewidth=0)
    sns.histplot(data=group[group['expertise_level'] == 'intermediate'],
                 x='episode_reward', color='orange', label='Intermediate',
                 kde=True, stat="density", linewidth=0)
    sns.histplot(data=group[group['expertise_level'] == 'expert'],
                 x='episode_reward', color='green', label='Expert', kde=True,
                 stat="density", linewidth=0)

    plt.title(f'Episode Reward Distribution for Task: {task}')
    plt.xlabel('Episode Reward')  # Label for the x-axis
    plt.ylabel('Density')  # Label for the y-axis
    plt.legend()  # Show the legend
    plt.show()  # Display the plot


def get_expertise_levels():
  expertise_level_to_policy_paths = collections.defaultdict(dict)

  for domain, domain_group in dataset_df.groupby('domain'):
    print(f'Processing domain: {domain}')
    task_to_expertise_policy_paths = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for task, task_group in domain_group.groupby('task'):
      print(f'\tProcessing task: {task}')
      for expertise_level, expertise_level_group in task_group.groupby(
          'expertise_level'):
        print(f'\t\tProcessing expertise_level: {expertise_level}')
        chosen_policies = np.random.choice(
            expertise_level_group['path'].values,
            NUM_POLICIES_PER_EXPERTISE_LEVEL,
            replace=False
        )
        task_to_expertise_policy_paths[task][
          expertise_level] = chosen_policies.tolist()
    expertise_level_to_policy_paths[domain] = task_to_expertise_policy_paths

  # Example of what the structure would look like
  print(json.dumps(expertise_level_to_policy_paths, indent=2))


def get_training_sample_cost():
  # Merge with a tolerance of 4000 steps
  dataset_df = pd.merge_asof(policy_path_df_sorted,
                             all_df_sorted,
                             on='step',
                             by=['domain', 'task', 'algo', 'seed'],
                             direction='nearest',
                             tolerance=4000)
