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


def get_training_sample_cost():
  pass
