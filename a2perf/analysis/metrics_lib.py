import collections
import concurrent.futures
import functools
import glob
import json
import multiprocessing
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from absl import logging

DOMAIN_DISPLAY_NAME = {
    'quadruped_locomotion': 'Quadruped Locomotion',
    'circuit_training': 'Circuit Training',
    'web_navigation': 'Web Navigation',
}

TASK_DISPLAY_NAME = {
    'dog_pace': 'Dog Pace',
    'dog_trot': 'Dog Trot',
    'dog_spin': 'Dog Spin',
}

ALGO_DISPLAY_NAME = {
    'sac': 'SAC',
    'ppo': 'PPO',
    'dqn': 'DQN',
    'ddqn': 'DDQN',
}

METRIC_DISPLAY_NAME = {
    'Metrics/AverageReturn': 'Episodic Returns',
}


def format_func(value, tick_number):
  # Convert to integer if the value is effectively a whole number
  if value.is_integer():
    return f'{int(value)}'
  else:
    return f'{value}'


def load_tb_data(log_file, tags=None):
  tf.compat.v1.enable_eager_execution()

  if tags is None:
    tags = []

  # Initialize separate lists for steps, values, and timestamps for each tag
  data = {}
  for tag in tags:
    data[f'{tag}_Step'] = []
    data[f'{tag}_Value'] = []
    data[f'{tag}_Timestamp'] = []

  for event in tf.compat.v1.train.summary_iterator(log_file):
    if event.HasField('summary'):
      for value in event.summary.value:
        if value.tag in tags:
          if value.HasField('simple_value'):
            data_value = value.simple_value
          elif value.HasField('tensor'):
            # Parse tensor_content as a tensor and then extract its value
            tensor = tf.make_ndarray(value.tensor)
            data_value = tensor.item()
          else:
            raise ValueError(
                f'Value type not recognized for tag {value.tag}. Expected'
                f' simple_value or tensor, got {value.WhichOneof("value")}'
            )

          data[f'{value.tag}_Step'].append(event.step)
          data[f'{value.tag}_Value'].append(data_value)
          data[f'{value.tag}_Timestamp'].append(
              pd.to_datetime(event.wall_time, unit='s')
          )

  if all(len(data[f'{tag}_Step']) == 0 for tag in tags):
    return pd.DataFrame()  # Return an empty DataFrame if no data

    # Construct and return the DataFrame
  return pd.DataFrame(data)


def process_tb_event_dir(event_file_path, tags=None):
  tf.compat.v1.enable_eager_execution()

  log_base_dir = os.path.dirname(event_file_path)
  exp_split = event_file_path.split('/')

  if 'collect' in event_file_path:
    # Single collect job output
    if exp_split[-3] == 'summaries':
      indices = [-7, -8, -9, -10, -11]

    # Some jobs have multiple collect job outputs, so increase the indices
    if exp_split[-4] == 'summaries':
      indices = [-8, -9, -10, -11, -12]
  else:
    indices = [-4, -5, -6, -7, -8]

  details_segment, algo, task, experiment_number, domain = [
      exp_split[i] for i in indices
  ]
  seed = int(re.search(r'seed_(\d+)', details_segment).group(1))
  skill_level = re.search(r'skill_level_(\w+)', details_segment).group(1)

  logging.info(f'Processing log dir: {event_file_path}')
  logging.info(
      f'\tDomain: {domain}, Task: {task}, Algo: {algo}, Experiment Number:'
      f' {experiment_number}, Seed: {seed}, Skill Level: {skill_level}'
  )
  data_csv_path = os.path.join(log_base_dir, 'data.csv')

  if 1 == 0 and os.path.exists(data_csv_path):
    data = pd.read_csv(data_csv_path, on_bad_lines='skip')

    # We can't load timestamp from csv, so we need to convert that column from
    # string to datetime
    for tag in tags:
      data[f'{tag}_Timestamp'] = pd.to_datetime(data[f'{tag}_Timestamp'])
    logging.info(f'Loaded data from {data_csv_path}')
  else:
    data = load_tb_data(event_file_path, tags)
  if data.empty:
    logging.warning(f'No data found in {event_file_path}')
    return None

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


def process_codecarbon_csv(csv_file_path):
  df = pd.read_csv(csv_file_path, on_bad_lines='skip')
  exp_split = csv_file_path.split('/')

  # Process path to extract experiment details
  if 'collect' in csv_file_path:
    indices = [-6, -7, -8, -9, -10]
  else:
    indices = [-4, -5, -6, -7, -8]

  exp_name, algo, task, experiment, domain = [exp_split[i] for i in indices]
  seed = re.search(r'seed_(\d+)', exp_name).group(1)
  skill_level = re.search(r'skill_level_(\w+)', exp_name).group(1)

  logging.info('Processing Experiment: %s, Seed: %s, Algo: %s', experiment,
               seed, algo)
  df['seed'] = int(seed)
  df['experiment'] = experiment
  df['algo'] = algo
  df['task'] = task
  df['domain'] = domain
  df['skill_level'] = skill_level

  # Convert timestamps and identify corrupt data
  df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
  corrupt_rows = df[df['timestamp'].isna()]

  if not corrupt_rows.empty:
    logging.warning('Corrupt rows due to invalid timestamps:')
    logging.warning(corrupt_rows)

  # Remove rows with corrupt timestamps
  df = df.dropna(subset=['timestamp'])
  df['timestamp'] = df['timestamp'].apply(
      lambda x: x.replace(tzinfo=None) if x.tzinfo else x
  )

  # Identify and handle corrupt rows for specific columns
  for tag in ['gpu_power', 'cpu_power', 'duration']:
    # Convert to numeric, coercing errors to NaN
    df[tag] = pd.to_numeric(df[tag], errors='coerce')

    # Log and drop rows where conversion failed (NaN values present)
    corrupt_rows = df[df[tag].isna()]
    if not corrupt_rows.empty:
      logging.warning(f'Corrupt rows due to invalid {tag}:')
      logging.warning(corrupt_rows)

    # Drop rows with NaN values in these columns
    df = df.dropna(subset=[tag])

  # Sort by timestamp
  df = df.sort_values(by='timestamp')

  return df


def correct_cpu_energy(df, cpus_per_collect_job=96,
    total_cpus_on_collect_machine=128, cpus_per_train_job=96,
    total_cpus_on_train_machine=96, true_cpu_tdp=120):
  # Reset the DataFrame index to ensure uniqueness
  df = df.reset_index(drop=True)

  # Create a job type column with either `collect` or `train` depending on whether gpu_energy is 0 or not
  df['job_type'] = 'collect'
  df.loc[df['gpu_power'] > 0, 'job_type'] = 'train'

  # Compute the actual CPU power based on the job type
  df['actual_cpu_power'] = 0
  collect_mask = df['job_type'] == 'collect'
  train_mask = df['job_type'] == 'train'

  df.loc[
    collect_mask, 'actual_cpu_power'] = true_cpu_tdp * cpus_per_collect_job / total_cpus_on_collect_machine
  df.loc[
    train_mask, 'actual_cpu_power'] = true_cpu_tdp * cpus_per_train_job / total_cpus_on_train_machine

  # Compute the cpu energy in kWh
  df['cpu_energy'] = (df['actual_cpu_power'] * df['duration'] / 3600) / 1000

  return df


def format_func(value, tick_number):
  # Function to format tick labels
  if value.is_integer():
    return f'{int(value)}'
  else:
    return f'{value}'


def downsample_steps(group, tag, n_steps=1000):
  """ Select a subset of steps at regular intervals that have sufficient data points across seeds """
  # Count the number of values at each step
  step_counts = group.groupby(f'{tag}_Step').size()

  # Filter steps with more than 3 values (for mean and std calculation)
  valid_steps = step_counts[step_counts > 2].index

  # Calculate the interval at which to select steps
  interval = max(1, len(valid_steps) // n_steps)

  # Select steps at regular intervals
  selected_steps = valid_steps[::interval]

  # Return the filtered group
  return group[group[f'{tag}_Step'].isin(selected_steps)]


def plot_training_reward_data(metrics_df,
    event_file_tags=('Metrics/AverageReturn',)):
  for tag in event_file_tags:
    metrics_df[f'{tag}_Duration_minutes'] = metrics_df[f'{tag}_Duration'] // 60
    tag_display_val = METRIC_DISPLAY_NAME.get(tag, tag)
    plot_df = metrics_df.groupby(['domain', 'task'])

    for (domain, task), group_df in plot_df:
      fig, ax = plt.subplots(1, 1, figsize=(16, 10))
      for algo in group_df['algo'].unique():
        group = group_df[group_df['algo'] == algo]
        group = downsample_steps(group=group, n_steps=750, tag=tag)

        # Plot for 'Step'
        sns.lineplot(
            x=f'{tag}_Step',
            y=f'{tag}_Value',
            data=group,
            label=f'{ALGO_DISPLAY_NAME.get(algo, algo)}'
        )

      min_max_steps = group_df.groupby('algo')[f'{tag}_Step'].agg(
          ['min', 'max'])
      common_max_step = min_max_steps[
        'max'].min()  # Use the minimum of the maximum steps across algos

      ax.set_xlim(0, common_max_step)
      ax.set_xlabel('Train Step')
      ax.set_ylabel(tag_display_val)
      ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
      ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
      title = f'{DOMAIN_DISPLAY_NAME.get(domain, domain)} - {TASK_DISPLAY_NAME.get(task, task)} (Train Steps)'
      ax.set_title(title)
      ax.legend()
      plt.tight_layout()
      plt.show()

      fig, ax = plt.subplots(1, 1, figsize=(16, 10))
      for algo in group_df['algo'].unique():
        group = group_df[group_df['algo'] == algo]
        group = downsample_steps(group=group, n_steps=750, tag=tag)

        sns.lineplot(
            x=f'{tag}_Duration_minutes',
            y=f'{tag}_Value',
            data=group,
            label=f'{ALGO_DISPLAY_NAME.get(algo, algo)}'
        )

      min_max_durations = group_df.groupby('algo')[
        f'{tag}_Duration_minutes'].agg(
          ['min', 'max'])
      common_max_duration = min_max_durations[
        'max'].min()  # Use the minimum of the maximum durations across algos

      ax.set_xlim(0, common_max_duration)
      ax.set_xlabel('Duration (minutes)')
      ax.set_ylabel(tag_display_val)
      ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
      ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
      title = f'{DOMAIN_DISPLAY_NAME.get(domain, domain)} - {TASK_DISPLAY_NAME.get(task, task)} (Duration)'
      ax.set_title(title)
      ax.legend()
      plt.tight_layout()
      plt.show()


def glob_path(path):
  return glob.glob(path, recursive=True)


def load_data(patterns):
  with multiprocessing.Pool() as pool:
    files = pool.map(
        glob_path,
        [
            pattern
            for pattern in patterns
        ],
    )
    pool.close()
    pool.join()
  files = [item for sublist in files for item in sublist]
  files = set(files)
  return files


def load_training_reward_data(
    base_dir, experiment_ids, event_file_tags=('Metrics/AverageReturn',)
):
  patterns = [
      os.path.join(base_dir, f'{exp_id}/**/collect/**/*events.out.tfevents*')
      for exp_id in experiment_ids
  ]
  event_log_dirs = load_data(patterns)
  logging.info(f'Found {len(event_log_dirs)} event log dirs')

  process_log_dir_fn = functools.partial(
      process_tb_event_dir, tags=event_file_tags
  )

  all_dfs = []
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for data in executor.map(process_log_dir_fn, event_log_dirs):
      if data is not None:
        logging.info('Processing log dir: %s',
                     f' {data.iloc[0]["domain"]}/{data.iloc[0]["task"]}/{data.iloc[0]["algo"]}/{data.iloc[0]["experiment"]}/{data.iloc[0]["seed"]}')

        all_dfs.append(data)
  metrics_df = pd.concat(all_dfs)
  logging.info('Loaded %s rows of data', len(metrics_df))

  # Get the number of seeds for each algo, domain, task combo to make sure
  # we have the right number of seeds
  seed_counts = metrics_df.groupby(['domain', 'task', 'algo']).seed.nunique()
  logging.info('Seed counts: %s', seed_counts)

  # Get the number of rows for each algo, domain, task combo to make sure
  # each experiment has the same number of steps
  row_counts = metrics_df.groupby(['domain', 'task', 'algo'])['seed'].count()
  logging.info('Row counts: %s', row_counts)

  # Since we have parallelized, distributed experiments, we'll see
  # the same 'Step' multiple times. We simply need to combine the values
  for tag in event_file_tags:
    value_col = f'{tag}_Value'
    step_col = f'{tag}_Step'

    # Define aggregation methods: mean for the value column, first for others
    aggregation = {value_col: 'mean'}
    for col in metrics_df.columns:
      if col not in [
          value_col,
          step_col,
          'domain',
          'task',
          'algo',
          'experiment',
          'seed',
      ]:
        aggregation[col] = 'first'

    # Group by and apply the specified aggregation
    df = (
        metrics_df.groupby(
            ['domain', 'task', 'algo', 'experiment', 'seed', step_col]
        )
        .agg(aggregation)
        .reset_index()
    )
    metrics_df = df

  row_counts = metrics_df.groupby(['domain', 'task', 'algo'])['seed'].count()
  logging.info('Row counts after removing duplicate steps: %s', row_counts)

  # Add a "duration" column to the DataFrame
  for tag in event_file_tags:
    # Flat column names for timestamp and duration
    timestamp_col = f'{tag}_Timestamp'
    duration_col = f'{tag}_Duration'

    # Calculate the Duration and assign it to the DataFrame
    metrics_df[duration_col] = metrics_df.groupby(
        ['domain', 'task', 'algo', 'experiment', 'seed']
    )[timestamp_col].transform(lambda x: x - x.min())

    # Convert duration to seconds and round to the nearest second
    metrics_df[duration_col] = (
        metrics_df[duration_col].dt.total_seconds().round().astype(int)
    )

  return metrics_df


def load_training_system_data(base_dir, experiment_ids):
  patterns = [
      os.path.join(base_dir, f'{exp_id}/**/train_emissions.csv')
      for exp_id in experiment_ids
  ]
  csv_files = load_data(patterns)
  logging.info('Found %s csv files', len(csv_files))

  all_dfs = []
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for data in executor.map(process_codecarbon_csv, csv_files):
      if data is not None:
        all_dfs.append(data)
  metrics_df = pd.concat(all_dfs)
  logging.info('Loaded %s rows of data', len(metrics_df))

  # return metrics_df
  return correct_cpu_energy(metrics_df)


def load_inference_metric_data(base_dir, experiment_ids):
  patterns = [
      os.path.join(base_dir, f'{exp_id}/**/inference_metrics_results.json')
      for exp_id in experiment_ids]
  json_files = load_data(patterns)
  logging.info('Found %s json files', len(json_files))

  # Load all of the json files
  all_dfs = []
  for json_file in json_files:
    logging.info('Processing json file: %s', json_file)
    indices = [-4, -5, -6, -7, -8]
    exp_split = json_file.split('/')
    details_segment, algo, task, experiment_number, domain = [
        exp_split[i] for i in indices
    ]
    seed = int(re.search(r'seed_(\d+)', details_segment).group(1))
    skill_level = re.search(r'skill_level_(\w+)', details_segment).group(1)

    logging.info('Processing log dir: %s', json_file)
    logging.info(
        '\tDomain: %s, Task: %s, Algo: %s, Experiment Number: %s, Seed: %s, Skill Level: %s',
        domain, task, algo, experiment_number, seed, skill_level)

    with open(json_file, 'r') as f:
      data = json.load(f)
      data_df = pd.DataFrame.from_dict(data, orient='index').reset_index()
      data_df = data_df.rename(columns={'index': 'metric'})
      # Add columns for domain/algo/task/expeirment/seed so we can group by them
      # later

      data_df['domain'] = domain
      data_df['task'] = task
      data_df['algo'] = algo
      data_df['experiment'] = experiment_number
      data_df['seed'] = seed
      data_df['skill_level'] = skill_level
      all_dfs.append(data_df)

  metrics_df = pd.concat(all_dfs)
  metrics = collections.defaultdict(dict)

  for metric, df in metrics_df.groupby('metric'):
    for (domain, task, algo,), group in df.groupby(
        ['domain', 'task', 'algo', ]
    ):
      # Each row has a list object in the 'values' column. We need to aggregate
      # these lists to get the mean and standard deviation
      all_values = []
      for values in group['values']:
        all_values.extend(values)

      mean_val = np.mean(all_values)
      std_val = np.std(all_values)
      metrics[metric][(domain, algo, task)] = {
          'mean': mean_val,
          'std': std_val
      }
  return metrics, metrics_df


def load_inference_system_data(base_dir, experiment_ids):
  patterns = [
      os.path.join(base_dir, f'{exp_id}/**/inference_emissions.csv')
      for exp_id in experiment_ids
  ]
  csv_files = load_data(patterns)
  logging.info('Found %s csv files', len(csv_files))

  process_codecarbon_csv_fn = functools.partial(process_codecarbon_csv)

  all_dfs = []
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for data in executor.map(process_codecarbon_csv_fn, csv_files):
      if data is not None:
        all_dfs.append(data)
  metrics_df = pd.concat(all_dfs)
  logging.info('Loaded %s rows of data', len(metrics_df))

  return metrics_df
