import collections
import concurrent.futures
import functools
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy
import seaborn as sns
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from a2perf import analysis

_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_BASE_DIR = flags.DEFINE_string('base_dir',
                                '/home/ikechukwuu/workspace/rl-perf/logs',
                                'Base directory for logs.')
_EXPERIMENT_IDS = flags.DEFINE_list('experiment_ids', [94408569],
                                    'Experiment IDs to process.')


def _initialize_plotting():
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


def check_window_size(window_size):
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


def load_tb_data(log_file, tags=None):
  tf.compat.v1.enable_eager_execution()

  if tags is None:
    tags = []

  # Initialize a nested dictionary for steps and values
  data = {(tag, 'Step'): [] for tag in tags}
  data.update({(tag, 'Value'): [] for tag in tags})
  data.update({(tag, 'Timestamp'): [] for tag in tags})

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
                f'Value type not recognized for tag {value.tag}. '
                f'Expected simple_value or tensor, got {value.WhichOneof("value")}'
            )

          data[(value.tag, 'Step')].append(event.step)
          data[(value.tag, 'Value')].append(data_value)
          data[(value.tag, 'Timestamp')].append(
              pd.to_datetime(event.wall_time, unit='s'))

  if all(len(data[tag, 'Step']) == 0 for tag in tags):
    return pd.DataFrame()  # Return an empty DataFrame if no data

  # Construct and return the DataFrame
  return pd.DataFrame(data).sort_index(axis=1)


def process_tb_event_dir(event_file_path, tags=None):
  tf.compat.v1.enable_eager_execution()

  log_base_dir = os.path.dirname(event_file_path)
  exp_split = event_file_path.split('/')

  # Extract the relevant information from the path
  domain = exp_split[-11]
  experiment_number = int(exp_split[-10])
  task = exp_split[-9]
  algo = exp_split[-8]

  # Extracting details from a segment in the path
  details_segment = exp_split[-7]

  seed = int(re.search(r'seed_(\d+)', details_segment).group(1))
  skill_level = re.search(r'skill_level_(\w+)', details_segment).group(1)

  logging.info(f'Processing log dir: {event_file_path}')
  logging.info(
      f'\tDomain: {domain}, Task: {task}, Algo: {algo}, Experiment Number: {experiment_number}, Seed: {seed}, Skill Level: {skill_level}')
  data_csv_path = os.path.join(log_base_dir, 'data.csv')

  if 1 == 0 and os.path.exists(data_csv_path):
    data = pd.read_csv(data_csv_path)
    logging.info(f'Loaded data from {data_csv_path}')
  else:
    data = load_tb_data(
        event_file_path, tags)
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


def plot_training_reward_data(metrics_df,
    event_file_tags=('Metrics/AverageReturn',)):
  # Plot each event_file_tag once using "Step" as the x-axis
  # then "duration" as the x-axis
  for tag in event_file_tags:
    # Group by domain/task/algo and plot the tag
    plot_df = metrics_df.groupby(['domain', 'task', 'algo'])

    for (domain, task, algo), group in plot_df:
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
          15, 5))  # Create subplots: one row, two columns

      # Plot using 'Step' as x-axis
      sns.lineplot(ax=ax1, x=(tag, 'Step'), y=(tag, 'Value'), data=group,
                   label=f'{domain}/{task}/{algo}')
      ax1.set_xlabel('Step')
      ax1.set_ylabel(tag)
      ax1.set_title(f'Step-wise Plot for {domain}/{task}/{algo}')
      ax1.legend()

      # Plot using 'Timestamp' as x-axis
      sns.lineplot(ax=ax2, x=(tag, 'Duration'), y=(tag, 'Value'), data=group,
                   label=f'{domain}/{task}/{algo}')
      ax2.set_xlabel('Duration')
      ax2.set_ylabel(tag)
      ax2.set_title(f'Duration-wise Plot for {domain}/{task}/{algo}')
      ax2.legend()

      plt.tight_layout()  # Adjust subplots to fit into figure area.
      plt.show()


def load_training_reward_data(base_dir, experiment_ids,
    event_file_tags=('Metrics/AverageReturn',)):
  event_log_dirs = []
  for exp_id in experiment_ids:
    logs = glob.glob(
        os.path.join(base_dir,
                     f'**/*{exp_id}*/**/collect/**/*events.out.tfevents*'),
        recursive=True)
    event_log_dirs.extend(logs)

  logging.info(f'Found {len(event_log_dirs)} event log dirs')

  process_log_dir_fn = functools.partial(process_tb_event_dir,
                                         tags=event_file_tags)

  all_dfs = []
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for data in executor.map(process_log_dir_fn, event_log_dirs):
      if data is not None:
        # print logging message based on the log dir
        print(
            f'Processing log dir: {data.iloc[0]["domain"]}/{data.iloc[0]["task"]}/{data.iloc[0]["algo"]}/{data.iloc[0]["experiment"]}/{data.iloc[0]["seed"]}')
        all_dfs.append(data)
  metrics_df = pd.concat(all_dfs)
  logging.info(f'Loaded {len(metrics_df)} rows of data')

  # Get the number of seeds for each algo, domain, task combo to make sure
  # we have the right number of seeds
  seed_counts = metrics_df.groupby(['domain', 'task', 'algo']).seed.nunique()
  logging.info(f'Seed counts: {seed_counts}')

  # Get the number of rows for each algo, domain, task combo to make sure
  # each experiment has the same number of steps
  row_counts = metrics_df.groupby(['domain', 'task', 'algo'])['seed'].count()
  logging.info(f'Row counts: {row_counts}')

  # Add a "duration" column to the DataFrame
  for tag in event_file_tags:
    # Create a MultiIndex for the new 'Duration' column
    duration_index = pd.MultiIndex.from_tuples([(tag, 'Duration')])
    timestamp_index = pd.MultiIndex.from_tuples([(tag, 'Timestamp')])
    # Calculate the Duration and assign it to the DataFrame
    metrics_df[duration_index] = metrics_df.groupby(
        ['domain', 'task', 'algo', 'experiment', 'seed'])[
      timestamp_index].transform(lambda x: x - x.min())

    # Convert duration to seconds and round to the nearest second
    metrics_df[duration_index] = metrics_df[duration_index].map(
        lambda x: np.round(x.total_seconds()).astype(int))

  return metrics_df


def load_training_system_data(base_dir, experiment_ids):
  pass


def load_inference_reward_data(base_dir, experiment_ids):
  pass


def load_inference_system_data(base_Dir, experiment_ids):
  pass


def get_training_system_metrics(data_df):
  pass


def get_training_reward_metrics(data_df, tag, index):
  reliability_metrics = analysis.get_training_reliability_metrics(
      data_df=data_df, tag=tag, index=index)
  return reliability_metrics


def get_inference_reward_metrics(data_df):
  pass


def get_inference_system_metrics(data_df):
  pass


def main(_):
  tf.compat.v1.enable_eager_execution()

  np.random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)
  _initialize_plotting()

  training_reward_metrics_df = load_training_reward_data(
      base_dir=_BASE_DIR.value,
      experiment_ids=_EXPERIMENT_IDS.value)
  print(training_reward_metrics_df.head())
  training_reward_metrics = get_training_reward_metrics(
      data_df=training_reward_metrics_df, tag='Metrics/AverageReturn',
      index='Step')

  training_system_metrics_df = load_training_system_data(
      base_dir=_BASE_DIR.value,
      experiment_ids=_EXPERIMENT_IDS.value)
  print(training_system_metrics_df.head())

  training_system_metrics = get_training_system_metrics(
      data_df=training_reward_metrics_df)

  inference_reward_metrics_df = load_inference_reward_data(
      base_dir=_BASE_DIR.value,
      experiment_ids=_EXPERIMENT_IDS.value)
  print(inference_reward_metrics_df.head())
  inference_reward_metrics = get_inference_reward_metrics(
      data_df=inference_reward_metrics_df)

  inference_system_metrics_df = load_inference_system_data(
      base_dir=_BASE_DIR.value,
      experiment_ids=_EXPERIMENT_IDS.value)
  print(inference_system_metrics_df.head())
  inference_system_metrics = get_inference_system_metrics(
      data_df=inference_system_metrics_df)


if __name__ == '__main__':
  app.run(main)
