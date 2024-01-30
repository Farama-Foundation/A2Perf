import collections
import concurrent.futures
import functools
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tensorflow as tf
from absl import flags
from absl import logging
from absl import app

_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_BASE_DIR = flags.DEFINE_string('base_dir',
                                '/home/ikechukwuu/workspace/rl-perf/logs',
                                'Base directory for logs.')
_EXPERIMENT_IDS = flags.DEFINE_list('experiment_ids', [94408569],
                                    'Experiment IDs to process.')


def no_op():
  event_file_tags = ('Metrics/AverageReturn',)


  # In[9]:



  reward_metrics_df.head()

  # In[61]:

  # Now drop all rows where Metrics/AverageReturn is NaN
  reward_metrics_df = reward_metrics_df.dropna(subset=['Metrics/AverageReturn'])
  reward_metrics_df.head()

  # In[ ]:

  # Now do a similar for the system metric data

  # In[12]:

  # Make sure all_df is sorted by step for our analysis below
  all_df = all_df.sort_values(
      by=['domain', 'task', 'algo', 'seed',
          'step'])  # or 'time' instead of 'step'

  # In[13]:

  # for each algo, domain and task, get the number of seeds, but use pandas
  # groupby
  seed_counts = all_df.groupby(['algo', 'domain', 'task'])['seed'].nunique()
  print(seed_counts)

  row_counts = all_df.groupby(['algo', 'domain', 'task'])['seed'].count()
  print(row_counts)

  # ### Variations in Plotting

  # In[16]:

  def prepare_data_for_task(task):
    """
    Prepare data for a specific task.
    """
    # Create a copy of the dataframe slice to avoid SettingWithCopyWarning
    task_df = all_df[all_df['task'] == task].copy()
    task_df['step_to_plot'] = task_df['step'] * 4096
    return task_df

  # In[17]:

  def plot_data_steps(task_df, x_column, y_column, hue_column, task,
      save_prefix=''):
    """
    Plot data for a specific task.
    """
    plt.figure()

    algo_list = task_df[hue_column].unique()  # Get the unique algorithms
    algo_list = np.sort(algo_list)  # Sort the algorithms
    hue_order = algo_list.tolist()  # Convert to a list

    axes = sns.lineplot(x=x_column, y=y_column, hue=hue_column, data=task_df,
                        errorbar='sd', palette='colorblind',
                        hue_order=hue_order)

    for line in axes.lines:
      line.set_alpha(0.75)

    axes.set_xlabel('Training Step', fontsize=14)
    axes.set_ylabel('Episode Reward', fontsize=14)

    task_title = task.replace('_', ' ').title()
    axes.set_title(f'Algorithm Comparison for Task: {task_title}', fontsize=16)
    axes.legend(title='Algorithm', loc='upper left', fontsize='medium')
    sns.despine()

    # Save the plot based on the task and any other identifiable information
    plt.savefig(f'{save_prefix}_{task}.png', dpi=300)
    plt.show()

  # In[18]:

  # Parallel data preparation
  tasks = all_df['task'].unique()
  prepared_data = []

  with concurrent.futures.ProcessPoolExecutor() as executor:
    for task_df in executor.map(prepare_data_for_task, tasks):
      prepared_data.append((task_df, task_df['task'].iloc[0]))

  # Sequential plotting
  for task_df, task in prepared_data:
    plot_data_steps(task_df, 'step_to_plot', 'episode_reward', 'algo', task,
                    save_prefix='steps')

  # In[19]:

  #  Subtract minimum timestamp for each group
  all_df['timestamp_adjusted'] = all_df.groupby(
      ['domain', 'experiment', 'algo', 'task', 'seed'])['timestamp'].transform(
      lambda x: x - x.min())

  # In[20]:

  # Convert to seconds and round to nearest second
  all_df['timestamp_adjusted'] = all_df[
    'timestamp_adjusted'].dt.total_seconds().round().astype(np.int64)

  # In[21]:

  # Let's examine a single group to make sure the timestamps make sense. Choose the group with seed=303, task=dog_pace, algo=PPO, experiment=2223
  seed_df = all_df[
    (all_df['seed'] == 303) & (all_df['task'] == 'dog_pace') & (
        all_df['algo'] == 'PPO') & (all_df['experiment'] == 2223)]

  # In[22]:

  # sort seed_df by timestamp_adjusted. if  we sort timestamp_adjusted, then step should be sorted as well
  seed_df = seed_df.sort_values(by='timestamp_adjusted')
  seed_df

  # In[23]:

  def prepare_data_for_task(task):
    """
    Prepare data for a specific task.
    """
    # Create a copy of the dataframe slice to avoid SettingWithCopyWarning
    task_df = all_df[all_df['task'] == task]
    return task_df

  # In[24]:

  def plot_data_duration(task_df, x_column, y_column, hue_column, task,
      save_prefix=''):
    """
    Plot data for a specific task.
    """
    plt.figure()
    algo_list = task_df[hue_column].unique()  # Get the unique algorithms
    algo_list = np.sort(algo_list)  # Sort the algorithms
    hue_order = algo_list.tolist()  # Convert to a list

    axes = sns.lineplot(x=x_column, y=y_column, hue=hue_column, data=task_df,
                        errorbar='sd', palette='colorblind',
                        hue_order=hue_order)
    for line in axes.lines:
      line.set_alpha(0.75)

    max_timestamps = task_df.groupby(['algo', 'seed'])[
      'timestamp_adjusted'].max().reset_index()
    min_of_max_timestamps = max_timestamps['timestamp_adjusted'].min()

    axes.set_xlim(0, min_of_max_timestamps)
    axes.set_xlabel('Duration (seconds)', fontsize=14)
    axes.set_ylabel('Episode Reward', fontsize=14)
    task_title = task.replace('_', ' ').title()

    axes.set_title(f'Algorithm Comparison for Task: {task_title}', fontsize=16)
    axes.legend(title='Algorithm', loc='upper left', fontsize='medium')
    sns.despine()

    plt.savefig(f'{save_prefix}_{task}.png', dpi=300)
    plt.show()

  # In[25]:

  # Parallel data preparation
  tasks = all_df['task'].unique()
  prepared_data = []

  with concurrent.futures.ProcessPoolExecutor() as executor:
    for task_df in executor.map(prepare_data_for_task, tasks):
      prepared_data.append((task_df, task_df['task'].iloc[0]))

  for task_df, task in prepared_data:
    plot_data_duration(task_df, 'timestamp_adjusted', 'episode_reward', 'algo',
                       task,
                       save_prefix='duration')

  # ### Determining the score thresholds for difficulties

  # In[26]:

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

  # In[27]:

  group_columns = ['domain', 'task']
  score_bounds = calculate_score_bounds_per_task(all_df, group_columns,
                                                 'episode_reward')
  score_bounds

  # In[28]:

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

  # In[29]:

  # Assuming all_df is your DataFrame and score_bounds contains the bounds per task
  all_df['expertise_level'] = all_df.apply(assign_expertise_level, axis=1,
                                           bounds=score_bounds)

  # In[30]:

  # Now plot histograms of the episode_rewards for each TASK, but change the color based on the expertise level
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

  # ### Use the saved checkpoints to select policies for data generation

  # In[31]:

  all_policy_paths = []
  for domain, experiments in all_experiments.items():
    for algo, exp_dict in experiments.items():
      for exp_id, task_dict in exp_dict.items():
        for task, seeds in task_dict.items():
          for seed in seeds:
            experiment_path = os.path.join(base_dir, domain,
                                           'debug' if debug else '',
                                           task, algo.lower(), str(exp_id),
                                           f'**/policies/*_steps.zip')
            policies_paths = glob.glob(experiment_path, recursive=True)
            print(f'Experiment Path: {experiment_path}')
            print(f'Number of policies: {len(policies_paths)}')

            all_policy_paths.extend(policies_paths)
  len(all_policy_paths)

  # In[32]:

  all_policy_paths = set(all_policy_paths)

  # In[33]:

  policy_path_df = pd.DataFrame(all_policy_paths, columns=['path'])
  policy_path_df.head()

  # In[34]:

  policy_path_df['step'] = policy_path_df['path'].str.extract(
      r'(\d+)_steps.zip', expand=False).astype(np.int64)
  policy_path_df['step'] = policy_path_df['step'] // 4096
  policy_path_df['step'] = policy_path_df['step'].astype(np.int64)
  policy_path_df['seed'] = policy_path_df['path'].str.extract(r'seed_(\d+)',
                                                              expand=False).astype(
      np.int64)

  policy_path_df['domain'] = 'quadruped_locomotion'
  policy_path_df['task'] = policy_path_df['path'].str.extract(
      r'quadruped_locomotion/(\w+)/', expand=False)
  policy_path_df['algo'] = policy_path_df['path'].str.extract(
      r'quadruped_locomotion/\w+/(\w+)/', expand=False)
  # capitalise the algo
  policy_path_df['algo'] = policy_path_df['algo'].str.upper()
  policy_path_df.head()

  # In[35]:

  policy_path_df_sorted = policy_path_df.sort_values('step')
  all_df_sorted = all_df[['domain', 'task', 'algo', 'seed', 'step',
                          'episode_reward', 'expertise_level']].sort_values(
      'step')

  # In[36]:

  # Merge with a tolerance of 4000 steps
  dataset_df = pd.merge_asof(policy_path_df_sorted,
                             all_df_sorted,
                             on='step',
                             by=['domain', 'task', 'algo', 'seed'],
                             direction='nearest',
                             tolerance=4000)
  dataset_df.head()

  # In[37]:

  len(dataset_df)

  # In[51]:

  NUM_POLICIES_PER_EXPERTISE_LEVEL = 50

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

  # In[52]:

  # json dumps for quadruped locomotion
  print(
      json.dumps(expertise_level_to_policy_paths['quadruped_locomotion'],
                 indent=2))

  # ## Computing the Training Sample Cost
  #

  # ## Training Metrics

  # In[ ]:

  EVAL_POINTS_PER_WINDOW = 5

  all_training_metrics = collections.defaultdict(dict)

  # In[ ]:

  for domain, domain_group in all_df.groupby('domain'):
    print(f'Processing domain: {domain}')
    for task, task_group in domain_group.groupby('task'):
      print(f'\tProcessing task: {task}')
      for algo, algo_group in task_group.groupby('algo'):
        print(f'\t\tProcessing algo: {algo}')
        for exp_id, exp_group in algo_group.groupby('experiment'):
          for seed, seed_group in exp_group.groupby('seed'):
            seed_df = seed_group.sort_values(by='step').copy()
            median_step_diff = seed_df['step'].diff().median()
            window_size = int(EVAL_POINTS_PER_WINDOW * median_step_diff)
            eval_points = list(
                range(np.ceil(window_size / 2).astype(int),
                      max(all_df['step']), int(median_step_diff)))
            print(
                f'Domain: {domain}, Algo: {algo}, Experiment: {exp_id}, Task: {task}, Seed: {seed}')
            print(f'\tMedian step difference: {median_step_diff}')
            print(f'\tWindow size: {window_size}')
            print()
            all_training_metrics[(domain, algo, exp_id, task, seed)][
              'eval_points'] = eval_points
            all_training_metrics[(domain, algo, exp_id, task, seed)][
              'window_size'] = window_size
            all_training_metrics[(domain, algo, exp_id, task, seed)][
              'median_step_diff'] = median_step_diff

  # ### Computing Reliability Metrics

  # In[ ]:

  window_fn = scipy.stats.iqr

  # In[ ]:

  for domain, domain_group in all_df.groupby('domain'):
    print(f'Processing domain: {domain}')
    for task, task_group in domain_group.groupby('task'):
      print(f'\tProcessing task: {task}')
      for algo, algo_group in task_group.groupby('algo'):
        print(f'\t\tProcessing algo: {algo}')
        all_iqr_values = []
        for exp_id, exp_group in algo_group.groupby('experiment'):
          for seed, seed_group in exp_group.groupby('seed'):
            seed_group['episode_reward_diff'] = seed_group[
              'episode_reward'].diff()
            seed_group = seed_group[['step', 'episode_reward_diff']].copy()
            seed_group = seed_group.dropna()
            steps, episode_reward_diff = seed_group.to_numpy().T

            window_size = \
              all_training_metrics[(domain, algo, exp_id, task, seed)][
                'window_size']
            for eval_point in \
                all_training_metrics[(domain, algo, exp_id, task, seed)][
                  'eval_points']:
              low_end = np.ceil(eval_point - (window_size / 2))
              high_end = np.floor(eval_point + (window_size / 2))

              eval_points_above = steps >= low_end
              eval_points_below = steps <= high_end
              eval_points_in_window = np.logical_and(eval_points_above,
                                                     eval_points_below)
              valid_eval_points = np.nonzero(eval_points_in_window)[0]

              if len(valid_eval_points) == 0:
                break

              # Apply window_fn to get the IQR for the current window
              window_iqr = window_fn(episode_reward_diff[valid_eval_points])
              all_iqr_values.append(window_iqr)
        mean_iqr = np.mean(all_iqr_values)
        std_iqr = np.std(all_iqr_values)
        all_training_metrics[(domain, algo, task,)][
          'dispersion_within_runs'] = dict(
            mean=mean_iqr, std=std_iqr)
        print(
            f'Domain: {domain}, Task: {task}, Algo: {algo}, Mean IQR: {mean_iqr}, Std IQR: {std_iqr}')

  #
  # ##### Long-term risk across time
  # Long - term risk across time represents the propensity of the agent to crash after achieving a higher performance measure

  # In[ ]:

  def compute_drawdown(sequence):
    """Computes the drawdown for a sequence of numbers.

      The drawdown at time T is the decline from the highest peak occurring at or
      before time T. https://en.wikipedia.org/wiki/Drawdown_(economics).

      The drawdown is always non-negative. A larger (more positive) drawdown
      indicates a larger drop.

    Args:
      sequence: A numpy array.

    Returns:
      A numpy array of same length as the original sequence, containing the
        drawdown at each timestep.
    """
    peak_so_far = np.maximum.accumulate(sequence)
    return peak_so_far - sequence

  # In[ ]:

  alpha = 0.95

  for domain, domain_group in all_df.groupby('domain'):
    print(f'Processing domain: {domain}')
    for task, task_group in domain_group.groupby('task'):
      print(f'\tProcessing task: {task}')
      for algo, algo_group in task_group.groupby('algo'):
        print(f'\t\tProcessing algo: {algo}')
        all_drawdowns = []
        for exp_id, exp_group in algo_group.groupby('experiment'):
          for seed, seed_group in exp_group.groupby('seed'):
            # Compute the drawdowns
            episode_rewards = seed_group['episode_reward'].values
            drawdowns = compute_drawdown(episode_rewards)
            all_drawdowns.extend(drawdowns)

        # Get the bottom "alpha" percent of drawdowns (we use the 95th percentile to get the bottom 5% of drawdowns)
        top_alpha_percent = np.percentile(all_drawdowns, alpha * 100)
        all_drawdowns = np.array(all_drawdowns)

        # CVaR is the average of the bottom "alpha" percent of drawdowns
        cvar = np.mean(all_drawdowns[all_drawdowns >= top_alpha_percent])

        print(f'\t\t\tCVaR: {cvar}')

        # Add the cvar to the dictionary
        all_training_metrics[(domain, algo, task,)]['long_term_risk'] = cvar

  # ##### Short-term risk across time
  # Short - term risk across time represents how volatile the agent is from eval point to eval point

  # In[ ]:

  alpha = 0.05

  for domain, domain_group in all_df.groupby('domain'):
    print(f'Processing domain: {domain}')
    for task, task_group in domain_group.groupby('task'):
      print(f'\tProcessing task: {task}')
      for algo, algo_group in task_group.groupby('algo'):
        print(f'\t\tProcessing algo: {algo}')
        all_diffs = []
        for exp_id, exp_group in algo_group.groupby('experiment'):
          for seed, seed_group in exp_group.groupby('seed'):
            seed_df = seed_group.sort_values(by='step').copy()
            seed_df['episode_reward_diff'] = seed_df['episode_reward'].diff()
            seed_df = seed_df.dropna()

            episode_reward_diffs = seed_df['episode_reward_diff'].values
            all_diffs.extend(episode_reward_diffs)

        bottom_alpha_percent = np.percentile(all_diffs, alpha * 100,
                                             method='linear')

        # CVaR is the average of the bottom "alpha" percent of diffs
        all_diffs = np.array(all_diffs)
        cvar = np.mean(
            all_diffs[all_diffs <= bottom_alpha_percent])
        cvar = -cvar  # make it positive for easier interpretation
        print(f'\t\t\tCVaR: {cvar}')
        all_training_metrics[(domain, algo, task,)][
          'short_term_risk'] = cvar

  #
  # ##### Risk across runs
  # Risk across runs tells use how poor the final performance of the worst runs are

  # In[ ]:

  # Get the final episode reward for each seed
  final_episode_rewards = all_df.groupby(['domain', 'task', 'algo', 'seed'])[
    'episode_reward'].last().reset_index()
  final_episode_rewards

  # In[ ]:

  # Now we just need to compute the CVaR of the final episode rewards
  alpha = 0.05

  # all experiments and seeds within a specific domain/task/algo should be grouped together
  for domain, domain_group in final_episode_rewards.groupby('domain'):
    for task, task_group in domain_group.groupby('task'):
      for algo, algo_group in task_group.groupby('algo'):
        # print(algo_group)  # this is the group we want to compute the CVaR for

        # Get the bottom "alpha" percent of final episode rewards
        rewards = algo_group['episode_reward'].values
        bottom_alpha_percent = np.percentile(rewards, alpha * 100,
                                             method='linear')
        cvar = np.mean(rewards[rewards <= bottom_alpha_percent])

        print(f'Processing domain: {domain}, task: {task}, algo: {algo}')
        print(f'\tCVaR: {cvar}')
        all_training_metrics[(domain, algo, task,)][
          'risk_across_runs'] = cvar

  # #### Dispersion across runs
  #

  # In[ ]:

  def lowpass_filter(curve, lowpass_thresh):
    filt_b, filt_a = scipy.signal.butter(8, lowpass_thresh)

    def butter_filter_fn(c):
      padlen = min(len(c) - 1, 3 * max(len(filt_a), len(filt_b)))
      return scipy.signal.filtfilt(filt_b, filt_a, curve, padlen=padlen)

    processed_curve = butter_filter_fn(curve)
    return processed_curve

  # In[ ]:

  # Plot the first curve before and after lowpass filtering
  # curve = ep_rew_clean[ep_rew_clean['seed'] == '37']['value'].values
  # lowpass_curve = lowpass_filter(curve, lowpass_thresh=0.01)

  ppo_curve = all_df[(all_df['algo'] == 'PPO') & (all_df['seed'] == 303) & (
      all_df['task'] == 'dog_pace')][
    'episode_reward'].values

  ddpg_curve = all_df[(all_df['algo'] == 'DDPG') & (all_df['seed'] == 303) & (
      all_df['task'] == 'dog_pace')][
    'episode_reward'].values

  low_pass_ppo_curve = lowpass_filter(ppo_curve, lowpass_thresh=0.01)
  low_pass_ddpg_curve = lowpass_filter(ddpg_curve, lowpass_thresh=0.01)

  # plot both curves with the original on the left, and lowpass on the right
  fig, axes = plt.subplots(1, 2, figsize=(20, 6))
  axes[0].plot(ppo_curve)
  axes[0].plot(ddpg_curve)
  axes[0].set_title('Original')
  axes[1].plot(low_pass_ppo_curve)
  axes[1].plot(low_pass_ddpg_curve)
  axes[1].set_title('Lowpass Filtered')

  # add a legend so it's easy to read
  axes[0].legend(['PPO', 'DDPG'])
  axes[1].legend(['PPO', 'DDPG'])

  # In[ ]:

  def apply_lowpass(curve):
    # Apply the lowpass filter directly on the curve
    low_pass_curve = lowpass_filter(curve, lowpass_thresh=0.01)
    return low_pass_curve

  # In[ ]:

  # Apply the function to the 'episode_reward' column of each group and assign the result to a new column
  all_df['lowpass_episode_reward'] = \
    all_df.groupby(['domain', 'task', 'algo', 'experiment', 'seed'])[
      'episode_reward'].transform(apply_lowpass)
  all_df.keys()

  # In[ ]:

  def compute_dispersion(curve):
    # Compute the dispersion as the IQR of the curve
    return scipy.stats.iqr(curve)

  # In[ ]:

  # Group by 'domain', 'task', 'algo', and 'steps', then apply the IQR computation
  dispersion_df = all_df.groupby(['domain', 'algo', 'task', 'step'])[
    'lowpass_episode_reward'].apply(compute_dispersion).reset_index()

  # Renaming the column for clarity
  dispersion_df.rename(columns={'lowpass_episode_reward': 'iqr_dispersion'},
                       inplace=True)
  dispersion_df

  # In[ ]:

  # Now that we have the dispersion values for specific steps, we can compute the mean and std of the dispersion for each algo
  mean_dispersion = dispersion_df.groupby(['domain', 'algo', 'task'])[
    'iqr_dispersion'].mean()
  std_dispersion = dispersion_df.groupby(['domain', 'algo', 'task'])[
    'iqr_dispersion'].std()

  # In[ ]:

  # Display the means and stds nicely. Should be durations for domain/algo/task
  for domain in mean_dispersion.index.levels[0]:
    print(domain)
    for algo in mean_dispersion.index.levels[1]:
      print(f'\t{algo}')
      for task in mean_dispersion.loc[domain, algo].index:
        print(
            f'\t\tTask: {task}, Mean Dispersion: {mean_dispersion.loc[domain, algo, task]}, Std Dispersion: {std_dispersion.loc[domain, algo, task]}')
        all_training_metrics[(domain, algo, task)][
          'dispersion_across_runs'] = dict(
            mean=mean_dispersion.loc[domain, algo, task],
            std=std_dispersion.loc[domain, algo, task])

  # ## System Metrics

  # In[ ]:

  training_system_metric_dirs = []
  for domain, _ in all_df.groupby('domain'):
    for task, _ in all_df.groupby('task'):
      for algo, _ in all_df.groupby('algo'):
        for exp_id, _ in all_df.groupby('experiment'):
          for seed, _ in all_df.groupby('seed'):
            experiment_path = os.path.join(base_dir, domain,
                                           task, algo.lower(), str(exp_id),
                                           f'**/train_emissions.csv')
            print(f'Experiment Path: {experiment_path}')

            training_system_metric_dirs.extend(glob.glob(experiment_path,
                                                         recursive=True))
  len(training_system_metric_dirs)

  # In[ ]:

  training_system_metric_dirs = set(training_system_metric_dirs)
  len(training_system_metric_dirs)

  # In[ ]:

  training_system_metrics_df = []
  for metric_dir in training_system_metric_dirs:
    split_dir = metric_dir.split('/')
    domain = split_dir[6]
    exp_name = split_dir[-3]
    exp_name_split = exp_name.split('_')
    print(exp_name_split)
    seed = exp_name_split[-5]
    experiment = split_dir[-4]
    algo = split_dir[-5]
    task = split_dir[-6]
    algo = algo.upper()
    print(f'Processing Experiment: {experiment}, Seed: {seed}, Algo: {algo}')

    df = pd.read_csv(metric_dir)
    df['seed'] = int(seed)
    df['experiment'] = int(experiment)
    df['algo'] = algo
    df['task'] = task
    df['domain'] = domain
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(
        lambda x: x.replace(tzinfo=None) if x.tzinfo else x)
    training_system_metrics_df.append(df)
  training_system_metrics_df = pd.concat(training_system_metrics_df)
  training_system_metrics_df

  # In[ ]:

  # Get the last duration for each individual experiment
  last_durations = \
    training_system_metrics_df.groupby(
        ['domain', 'algo', 'task', 'experiment', 'seed'])[
      'duration'].last()

  # In[ ]:

  # Now find the mean and std of the last duration for each algo
  mean_last_duration = last_durations.groupby(['domain', 'algo', 'task']).mean()
  std_last_duration = last_durations.groupby(['domain', 'algo', 'task']).std()
  mean_last_duration

  # In[ ]:

  mean_last_duration.index.levels

  # In[ ]:

  # Display the means and stds nicely. Should be durations for domain/algo/task
  for domain in mean_last_duration.index.levels[0]:
    print(domain)
    for algo in mean_last_duration.index.levels[1]:
      print(f'\t{algo}')
      for task in mean_last_duration.loc[domain, algo].index:
        print(
            f'\t\tTask: {task}, Mean Last Duration: {mean_last_duration.loc[domain, algo, task]}, Std Last Duration: {std_last_duration.loc[domain, algo, task]}')
        all_training_metrics[(domain, algo, task)]['wall_clock_time'] = dict(
            mean=mean_last_duration.loc[domain, algo, task],
            std=std_last_duration.loc[domain, algo, task])

  # In[ ]:

  training_peak_ram_usage = \
    training_system_metrics_df.groupby(
        ['domain', 'algo', 'task', 'experiment', 'seed'])[
      'ram_process'].max()

  training_mean_peak_ram = training_peak_ram_usage.groupby(
      ['domain', 'algo', 'task']).mean()
  training_std_peak_ram = training_peak_ram_usage.groupby(
      ['domain', 'algo', 'task']).std()

  # Display the means and stds nicely
  for domain in training_mean_peak_ram.index.levels[0]:
    for algo in training_mean_peak_ram.index.levels[1]:
      for task in training_mean_peak_ram.index.levels[2]:
        all_training_metrics[(domain, algo, task)]['peak_ram_usage'] = dict(
            mean=training_mean_peak_ram.loc[domain, algo, task],
            std=training_std_peak_ram.loc[domain, algo, task])

        print(
            f'Domain: {domain}, Algo: {algo}, Task: {task}, Mean Peak RAM Usage: {training_mean_peak_ram.loc[domain, algo, task]}, Std Peak RAM Usage: {training_std_peak_ram.loc[domain, algo, task]}')

  # In[ ]:

  mean_training_ram_usage = \
    training_system_metrics_df.groupby(['domain', 'algo', 'task'])[
      'ram_process'].mean()
  std_training_ram_usage = \
    training_system_metrics_df.groupby(['domain', 'algo', 'task'])[
      'ram_process'].std()

  # Display the means and stds nicely
  for domain in mean_training_ram_usage.index.levels[0]:
    for algo in mean_training_ram_usage.index.levels[1]:
      for task in mean_training_ram_usage.index.levels[2]:
        all_training_metrics[(domain, algo, task)]['ram_usage'] = dict(
            mean=mean_training_ram_usage.loc[domain, algo, task],
            std=std_training_ram_usage.loc[domain, algo, task])

        print(
            f'Domain: {domain}, Algo: {algo}, Task: {task}, Mean RAM Usage: {mean_training_ram_usage.loc[domain, algo, task]}, Std RAM Usage: {std_training_ram_usage.loc[domain, algo, task]}')

  # ## Computing the Training Sample Cost
  #

  # In[ ]:

  # system_metrics_df['timestamp'] = pd.to_datetime(system_metrics_df['timestamp'])
  # pd.merge_asof(top_k_policies_df, system_metrics_df, left_on='timestamp',
  #               right_on='timestamp', direction='nearest', )

  # ---
  # # Inference Metrics
  #

  # In[ ]:

  all_inference_metrics = collections.defaultdict(dict)

  # ## Reliability Metrics

  # In[ ]:

  log_dirs = []

  for domain, domain_group in all_df.groupby('domain'):
    for algo, algo_group in domain_group.groupby('algo'):
      for exp_id, exp_group in algo_group.groupby('experiment'):
        for task, task_group in exp_group.groupby('task'):
          for seed, seed_group in task_group.groupby('seed'):
            experiment_path = os.path.join(base_dir, domain, task, algo.lower(),
                                           str(exp_id),
                                           f'**/inference_metrics_results.json')
            print(f'Experiment Path: {experiment_path}')

            log_dirs.extend(glob.glob(experiment_path, recursive=True))
  len(log_dirs)

  # In[ ]:

  log_dirs = set(log_dirs)
  len(log_dirs)

  # In[ ]:

  inference_df = []

  for log_dir in log_dirs:
    split_dir = os.path.normpath(log_dir).split(os.sep)
    exp_name = split_dir[-3]
    exp_name_split = exp_name.split('_')
    seed = exp_name_split[-5]
    experiment = split_dir[-4]
    algo = split_dir[-5]
    task = split_dir[-6]
    domain = split_dir[-7]

    print(
        f'Processing Experiment: {experiment}, Seed: {seed}, Domain: {domain}, Task: {task}, Algo: {algo}')

    data = json.load(open(log_dir))

    for metric, values in data.items():
      df = pd.DataFrame.from_dict(values['values'], orient='columns', )
      df['domain'] = domain
      df['task'] = task
      df['seed'] = int(seed)
      df['experiment'] = int(experiment)
      df['algo'] = algo
      df['algo'] = df['algo'].str.upper()
      df['metric'] = metric
      inference_df.append(df)
  inference_df = pd.concat(inference_df)
  inference_df

  # ### Risk Across Rollouts and Dispersion Across Rollouts

  # In[ ]:

  # Get the bottom "alpha" percent of rollouts (we use the 5th percentile to get the bottom 5% of rollouts)
  reliability_df = inference_df[inference_df['metric'] == 'rollout_returns']
  for domain, domain_group in reliability_df.groupby('domain'):
    print(f'Processing domain: {domain}')
    for task, task_group in domain_group.groupby('task'):
      print(f'\tProcessing task: {task}')
      for algo, algo_group in task_group.groupby('algo'):
        print(f'\t\tProcessing algo: {algo}')
        values = algo_group[0]

        # get the bottom 5% of rollouts
        bottom_alpha_percent = np.percentile(values, 5)

        # CVaR is the average of the bottom "alpha" percent of rollouts
        cvar = np.mean(values[values <= bottom_alpha_percent])
        print(f'\t\t\tCVaR: {cvar}')

        all_inference_metrics[(domain, algo, task,)][
          'risk_across_rollouts'] = cvar

        # Also get the dispersion across rollouts
        iqr = scipy.stats.iqr(values)
        print(f'\t\t\tIQR: {iqr}')
        all_inference_metrics[(domain, algo, task,)][
          'dispersion_across_rollouts'] = iqr

  # In[ ]:

  reliability_df

  # In[ ]:

  # Performance is actually a training metric, but we can compute it here using the rollout returns of final policies in realibility_df
  mean_performance = reliability_df.groupby(['domain', 'algo', 'task'])[
    0].mean()
  std_performance = reliability_df.groupby(['domain', 'algo', 'task'])[0].std()

  # In[ ]:

  # Display the means and stds nicely
  for domain in mean_performance.index.levels[0]:
    for algo in mean_performance.index.levels[1]:
      for task in mean_performance.index.levels[2]:
        print(
            f'Domain: {domain}, Algo: {algo}, Task: {task}, Mean Performance: {mean_performance.loc[domain, algo, task]}, Std Performance: {std_performance.loc[domain, algo, task]}')
        all_training_metrics[(domain, algo, task)]['returns'] = dict(
            mean=mean_performance.loc[domain, algo, task],
            std=std_performance.loc[domain, algo, task])

  # In[ ]:

  # Mean/STD of the inference time metric
  inference_time_df = inference_df[inference_df['metric'] == 'inference_time']
  mean_inference_time = inference_time_df.groupby(['domain', 'algo', 'task', ])[
    0].mean()
  std_inference_time = inference_time_df.groupby(['domain', 'algo', 'task', ])[
    0].std()
  mean_inference_time, std_inference_time

  # In[ ]:

  # Add results to the dict
  for domain, domain_group in mean_inference_time.groupby('domain'):
    for algo, algo_group in domain_group.groupby('algo'):
      for task, task_group in algo_group.groupby('task'):
        print(
            f'Domain: {domain}, Algo: {algo}, Task: {task}, Mean Inference Time: {mean_inference_time.loc[domain, algo, task]}, Std Inference Time: {std_inference_time.loc[domain, algo, task]}')
        all_inference_metrics[(domain, algo, task,)]['inference_time'] = dict(
            mean=mean_inference_time.loc[domain, algo, task],
            std=std_inference_time.loc[domain, algo, task])

  # ## System Metrics
  #

  # In[ ]:

  log_dirs = []

  for domain, domain_group in all_df.groupby('domain'):
    for algo, algo_group in domain_group.groupby('algo'):
      for exp_id, exp_group in algo_group.groupby('experiment'):
        for task, task_group in exp_group.groupby('task'):
          for seed, seed_group in task_group.groupby('seed'):
            experiment_path = os.path.join(base_dir, domain, task, algo.lower(),
                                           str(exp_id),
                                           f'**/inference_emissions.csv')
            print(f'Experiment Path: {experiment_path}')

            log_dirs.extend(glob.glob(experiment_path, recursive=True))
  len(log_dirs)

  # In[ ]:

  log_dirs = set(log_dirs)
  len(log_dirs)

  # In[ ]:

  inference_system_metrics_df = []
  for log_dir in log_dirs:
    split_dir = os.path.normpath(log_dir).split(os.sep)
    exp_name = split_dir[-3]
    exp_name_split = exp_name.split('_')
    seed = exp_name_split[-5]
    experiment = split_dir[-4]
    algo = split_dir[-5]
    task = split_dir[-6]
    domain = split_dir[-7]

    print(
        f'Processing Experiment: {experiment}, Seed: {seed}, Domain: {domain}, Algo: {algo}, Task: {task}')

    df = pd.read_csv(log_dir)
    df['domain'] = domain
    df['task'] = task
    df['seed'] = seed
    df['experiment'] = experiment
    df['algo'] = algo
    df['algo'] = df['algo'].str.upper()
    inference_system_metrics_df.append(df)
  inference_system_metrics_df = pd.concat(inference_system_metrics_df)
  inference_system_metrics_df

  # In[ ]:

  inference_peak_ram = \
    inference_system_metrics_df.groupby(
        ['domain', 'algo', 'task', 'experiment', 'seed'])[
      'ram_process'].max()

  inference_mean_peak_ram = inference_peak_ram.groupby(
      ['domain', 'algo', 'task']).mean()
  inference_std_peak_ram = inference_peak_ram.groupby(
      ['domain', 'algo', 'task']).std()

  # In[ ]:

  # Display the means and stds nicely
  for domain in inference_mean_peak_ram.index.levels[0]:
    for algo in inference_mean_peak_ram.index.levels[1]:
      for task in inference_mean_peak_ram.index.levels[2]:
        print(
            f'Domain: {domain}, Algo: {algo}, Task: {task}, Mean Peak RAM: {inference_mean_peak_ram.loc[domain, algo, task]}, Std Peak RAM: {inference_std_peak_ram.loc[domain, algo, task]}')
        all_inference_metrics[(domain, algo, task)]['peak_ram_usage'] = dict(
            mean=inference_mean_peak_ram.loc[domain, algo, task],
            std=inference_std_peak_ram.loc[domain, algo, task])

  # In[ ]:

  inference_mean_ram_usage = \
    inference_system_metrics_df.groupby(['domain', 'algo', 'task'])[
      'ram_process'].mean()
  inference_std_ram_usage = \
    inference_system_metrics_df.groupby(['domain', 'algo', 'task'])[
      'ram_process'].std()
  inference_mean_ram_usage, inference_std_ram_usage

  # In[ ]:

  for domain in inference_mean_ram_usage.index.levels[0]:
    for algo in inference_mean_ram_usage.index.levels[1]:
      for task in inference_mean_ram_usage.index.levels[2]:
        print(
            f'Domain: {domain}, Algo: {algo}, Task: {task}, Mean RAM Usage: {inference_mean_ram_usage.loc[domain, algo, task]}, Std RAM Usage: {inference_std_ram_usage.loc[domain, algo, task]}')
        all_inference_metrics[(domain, algo, task)]['ram_usage'] = dict(
            mean=inference_mean_ram_usage.loc[domain, algo, task],
            std=inference_std_ram_usage.loc[domain, algo, task])

  # # Display all metrics nicely

  # In[ ]:

  train_table = dict(
      category={
          "Application": ["returns"],
          "Reliability": ["dispersion_across_runs", "dispersion_within_runs",
                          "short_term_risk",
                          "long_term_risk", "risk_across_runs"],
          "System": ["peak_ram_usage", "ram_usage", "wall_clock_time"]
      }
      ,
      units={
          "Application": ['100 eps.'],
          "Reliability": ["IQR", "IQR", "CVaR", "CVaR", "CVaR"],
          "System": ["GB", "GB", "Hours"]
      })

  inference_table = dict(
      category={
          "Application": ['N/A'],
          "Reliability": ["dispersion_across_rollouts", "risk_across_rollouts"],
          "System": ["peak_ram_usage", "ram_usage", "inference_time"]
      },
      units={
          "Application": [''],
          "Reliability": ["IQR", "CVaR"],
          "System": ["GB", "GB", "ms"]
      }
  )

  criteria_dict = dict(
      returns='max',
      dispersion_across_runs='min',
      dispersion_within_runs='min',
      dispersion_across_rollouts='min',
      short_term_risk='min',
      long_term_risk='min',
      risk_across_runs='max',
      peak_ram_usage='min',
      ram_usage='min',
      wall_clock_time='min',
      inference_time='min'
  )

  # In[ ]:

  def format_number(num):
    if -1000 < num < 1000:
      return f"{num:.2f}"
    else:
      return f"{num:.2e}"

  def format_number_bold(num, is_bold):
    formatted_num = format_number(num)
    return f"\\textbf{{{formatted_num}}}" if is_bold else formatted_num

  def is_best(value, all_values, criteria):
    if criteria == 'max':
      return value == max(all_values)
    elif criteria == 'min':
      return value == min(all_values)
    return False

  def generate_latex_table_header(task_name, phase):
    header_title = task_name.replace('_', ' ').title()
    header = [
        f'\\multicolumn{{4}}{{|c|}}{{\\textbf{{{header_title} ({phase})}}}} \\\\',
        '\\hline',
        '\\textbf{Category} & \\textbf{Metric Name} & \\textbf{PPO} & \\textbf{DDPG} \\\\',
        '\\hline'
    ]
    return '\n'.join(header)

  def get_metric_values(metrics_dict, domain, task, metric, criteria):
    metric_values = {}
    all_values = []

    # Collect all mean values for comparison
    for algo in ["PPO", "DDPG"]:
      key = (domain, algo, task)
      if key in metrics_dict and metric in metrics_dict[key]:
        values = metrics_dict[key][metric]
        mean_val = values['mean'] if isinstance(values, dict) else values
        all_values.append(mean_val)

    # Determine the best value and format numbers
    for algo in ["PPO", "DDPG"]:
      key = (domain, algo, task)
      if key in metrics_dict and metric in metrics_dict[key]:
        values = metrics_dict[key][metric]
        mean_val = values['mean'] if isinstance(values, dict) else values
        std_val = values['std'] if isinstance(values,
                                              dict) and 'std' in values else None

        is_bold = is_best(mean_val, all_values, criteria)

        # Special handling for specific metrics
        if metric == 'wall_clock_time':
          mean_val /= 3600
          std_val /= 3600
        elif metric == 'inference_time':
          mean_val *= 1000
          std_val *= 1000

        mean_str = format_number_bold(mean_val, is_bold)
        if std_val:
          std_str = format_number_bold(std_val,
                                       is_bold)  # Bold std if mean is bold
          metric_values[algo] = f"{mean_str} $\\pm$ {std_str}"
        else:
          metric_values[algo] = f"{mean_str}"

      else:
        metric_values[algo] = "N/A"

    return metric_values

  def add_metric_rows_to_table(metrics_table, metrics_dict, domain, task,
      criteria_dict):
    latex_table_rows = []
    for category, metrics in metrics_table['category'].items():
      units = metrics_table['units']
      unit_list = units[category]
      first_metric = True
      for i, metric in enumerate(metrics):
        criteria = criteria_dict.get(metric,
                                     'max')  # Default to 'max' if not specified
        metric_values = get_metric_values(metrics_dict, domain, task, metric,
                                          criteria)
        metric_name = f'{metric.replace("_", " ").title()} ({unit_list[i]})'
        row = f'\t\t & {metric_name} & {metric_values["PPO"]} & {metric_values["DDPG"]} \\\\\n'
        if first_metric:
          row = f'\t\t\\multirow{{{len(metrics)}}}{{*}}{{{category}}}' + row
          first_metric = False
        latex_table_rows.append(row)
      latex_table_rows.append('\t\t\\hline\n')
    return ''.join(latex_table_rows)

  # In[ ]:

  for domain, domain_group in all_df.groupby('domain'):
    for task, task_group in domain_group.groupby('task'):
      header_title = task.replace('_', ' ').title()
      latex_table_string = ['\\begin{figure}[!htbp]',
                            '\\resizebox{1\\textwidth}{!}{',
                            '\\begin{tabular}{|c|l|c|c|}',
                            '\\hline', ]
      latex_table_string.append(generate_latex_table_header(task, "Training"))

      # Add training metric rows with criteria
      latex_table_string.append(
          add_metric_rows_to_table(train_table, all_training_metrics, domain,
                                   task, criteria_dict))

      # Add inference header
      latex_table_string.append(generate_latex_table_header(task, "Inference"))

      # Add inference metric rows with criteria
      latex_table_string.append(
          add_metric_rows_to_table(inference_table, all_inference_metrics,
                                   domain,
                                   task, criteria_dict))

      # Closing tags
      latex_table_string.extend(
          ['\t\\end{tabular}}\n',
           f'\t\\caption{{Training and Inference Metrics for {header_title}}}\n',
           f'\t\\label{{tab:appendix_metrics_{task}}}\n', '\\end{figure}\n'])

      latex_table = ''.join(latex_table_string)
      print(latex_table)
      print("\n\n")

  # # Sandbox

  # In[ ]:


def load_tb_data(event_file, tags):
  if tags is None:
    tags = []

  # Initialize data storage
  all_data = []

  # Use TFRecordDataset to read the event file
  raw_dataset = tf.data.TFRecordDataset(event_file)

  for raw_record in raw_dataset:
    event = tf.compat.v1.Event.FromString(raw_record.numpy())

    # If there is no summary field, immediately skip
    if not event.summary.value:
      continue

    # Convert the timestamp to a datetime object for future pandas use
    timestamp = pd.to_datetime(event.wall_time, unit='s')

    # Prepare a dictionary to hold the data for this event
    event
    event_data = {'step': event.step, 'timestamp': timestamp}
    for value in event.summary.value:
      if value.tag in tags:
        tensor_content = tf.io.decode_raw(value.tensor.tensor_content,
                                          tf.float32).numpy()
        event_data[value.tag] = tensor_content[0]

    all_data.append(event_data)

  # Create a DataFrame from the accumulated data with index 'step'
  data = pd.DataFrame(all_data).set_index('step')
  return data


def process_log_dir(log_dir, tags=None):
  log_base_dir = os.path.dirname(log_dir)
  exp_split = log_dir.split('/')

  # Extract the relevant information from the path
  domain = exp_split[-11]
  experiment_number = int(exp_split[-10])
  task = exp_split[-9]
  algo = exp_split[-8]

  # Extracting details from a segment in the path
  details_segment = exp_split[-7]

  seed = int(re.search(r'seed_(\d+)', details_segment).group(1))
  skill_level = re.search(r'skill_level_(\w+)', details_segment).group(1)

  logging.info(f'Processing log dir: {log_dir}')
  logging.info(
      f'\tDomain: {domain}, Task: {task}, Algo: {algo}, Experiment Number: {experiment_number}, Seed: {seed}, Skill Level: {skill_level}')
  data_csv_path = os.path.join(log_base_dir, 'data.csv')

  if os.path.exists(data_csv_path):
    data = pd.read_csv(data_csv_path)
    logging.info(f'Loaded data from {data_csv_path}')
  else:
    data = load_tb_data(
        log_dir, tags)

    # Add the experiment details to the DataFrame
    data['domain'] = domain
    data['task'] = task
    data['algo'] = algo
    data['experiment'] = experiment_number
    data['seed'] = seed
    data['skill_level'] = skill_level

    data.to_csv(data_csv_path)
    logging.info(f'Saved data to {data_csv_path}')
  return data


def compute_training_metrics(base_dir, experiment_ids,
    event_file_tags=('Metrics/AverageReturn',)):
  event_log_dirs = []
  for exp_id in experiment_ids:
    pattern = os.path.join(base_dir,
                           f'**/{exp_id}/**/*collect*/**/*events.out.tfevents*')
    event_log_dirs.extend(glob.glob(pattern, recursive=True))

  logging.info(f'Found {len(event_log_dirs)} event log dirs')

  process_log_dir_fn = functools.partial(process_log_dir,
                                         tags=event_file_tags)

  all_dfs = []
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for data in executor.map(process_log_dir_fn, event_log_dirs[:100]):
      if data is not None:
        # print logging message based on the log dir
        print(
            f'Processing log dir: {data.iloc[0]["domain"]}/{data.iloc[0]["task"]}/{data.iloc[0]["algo"]}/{data.iloc[0]["experiment"]}/{data.iloc[0]["seed"]}')
        all_dfs.append(data)
  reward_metrics_df = pd.concat(all_dfs)
  logging.info(f'Loaded {len(reward_metrics_df)} rows of data')


def compute_system_metrics(base_dir, experiment_ids):
  emission_log_dirs = []
  for exp_id in experiment_ids:
    pattern = os.path.join(base_dir,
                           f'**/{exp_id}/**/*collect*/**/train_emissions.csv')
    emission_log_dirs.extend(glob.glob(pattern, recursive=True))

  pass


def main(_):
  tf.compat.v1.enable_eager_execution()

  # Set some random seeds for reproducibility
  np.random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)

  # Set some defaults for seabron and matplotlib
  sns.set_style("whitegrid")
  plt.rcParams['figure.figsize'] = (12, 6)
  plt.rcParams['font.size'] = 14
  plt.rcParams['axes.labelsize'] = 14
  plt.rcParams['axes.titlesize'] = 16
  plt.rcParams['legend.fontsize'] = 12
  plt.rcParams['xtick.labelsize'] = 12
  plt.rcParams['ytick.labelsize'] = 12
  plt.rcParams['figure.titlesize'] = 16
  plt.rcParams['figure.dpi'] = 100
  plt.rcParams['savefig.dpi'] = 100
  plt.rcParams['savefig.format'] = 'png'
  plt.rcParams['savefig.bbox'] = 'tight'

  training_metrics = compute_training_metrics(base_dir=_BASE_DIR.value,
                                              experiment_ids=_EXPERIMENT_IDS.value)
  system_metrics = compute_system_metrics(base_dir=_BASE_DIR.value,
                                          experiment_ids=_EXPERIMENT_IDS.value)

  # Display the metrics nicely


if __name__ == '__main__':
  app.run(main)
