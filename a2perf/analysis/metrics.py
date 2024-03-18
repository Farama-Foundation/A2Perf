import concurrent.futures
import functools
import glob
import os
import re

from a2perf import analysis
from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_BASE_DIR = flags.DEFINE_string(
    'base_dir',
    '/home/ikechukwuu/workspace/rl-perf/logs',
    'Base directory for logs.',
)
_EXPERIMENT_IDS = flags.DEFINE_list(
    'experiment_ids', [94408569], 'Experiment IDs to process.'
)


def _initialize_plotting():
  sns.set_style('whitegrid')
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
    indices = [-7, -8, -9, -10, -11]
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
    data = pd.read_csv(data_csv_path)
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
  df = pd.read_csv(csv_file_path)
  exp_split = csv_file_path.split('/')

  # Process path to extract experiment details
  if 'collect' in csv_file_path:
    indices = [-6, -7, -8, -9, -10]
  else:
    indices = [-4, -5, -6, -7, -8]

  exp_name, algo, task, experiment, domain = [exp_split[i] for i in indices]
  seed = re.search(r'seed_(\d+)', exp_name).group(1)

  print(f'Processing Experiment: {experiment}, Seed: {seed}, Algo: {algo}')

  df['seed'] = int(seed)
  df['experiment'] = experiment
  df['algo'] = algo
  df['task'] = task
  df['domain'] = domain

  # Convert timestamps and identify corrupt data
  df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
  corrupt_rows = df[df['timestamp'].isna()]

  # Make sure all of the power/ram fields are floats
  # float_columns = [
  #     'ram_process', 'gpu_power', 'cpu_power', 'ram_power', 'gpu_energy',
  #     'ram_energy', 'cpu_energy']
  #
  # df[float_columns] = df[float_columns].apply(pd.to_numeric, errors='coerce')
  # corrupt_rows = pd.concat([corrupt_rows, df[df[float_columns].isna()]])
  #
  # Print corrupt rows
  if not corrupt_rows.empty:
    logging.warning('Corrupt rows due to invalid timestamps:')
    logging.warning(corrupt_rows)

  # Remove rows with corrupt timestamps
  df = df.dropna(subset=['timestamp'])
  df['timestamp'] = df['timestamp'].apply(
      lambda x: x.replace(tzinfo=None) if x.tzinfo else x
  )

  # Sort by timestamp
  df = df.sort_values(by='timestamp')
  return df


def plot_training_reward_data(
    metrics_df, event_file_tags=('Metrics/AverageReturn',)
):
  # Plot each event_file_tag once using "Step" as the x-axis
  # then "duration" as the x-axis
  for tag in event_file_tags:
    # Group by domain/task/algo and plot the tag
    plot_df = metrics_df.groupby(['domain', 'task', 'algo'])

    for (domain, task, algo), group in plot_df:
      fig, (ax1, ax2) = plt.subplots(
          1, 2, figsize=(15, 5)
      )  # Create subplots: one row, two columns

      # Plot using 'Step' as x-axis
      sns.lineplot(
          ax=ax1,
          x=(tag, 'Step'),
          y=(tag, 'Value'),
          data=group,
          label=f'{domain}/{task}/{algo}',
      )
      ax1.set_xlabel('Step')
      ax1.set_ylabel(tag)
      ax1.set_title(f'Step-wise Plot for {domain}/{task}/{algo}')
      ax1.legend()

      # Plot using 'Timestamp' as x-axis
      sns.lineplot(
          ax=ax2,
          x=(tag, 'Duration'),
          y=(tag, 'Value'),
          data=group,
          label=f'{domain}/{task}/{algo}',
      )
      ax2.set_xlabel('Duration')
      ax2.set_ylabel(tag)
      ax2.set_title(f'Duration-wise Plot for {domain}/{task}/{algo}')
      ax2.legend()

      plt.tight_layout()  # Adjust subplots to fit into figure area.
      plt.show()


def load_training_reward_data(
    base_dir, experiment_ids, event_file_tags=('Metrics/AverageReturn',)
):
  event_log_dirs = []
  for exp_id in experiment_ids:
    logs = glob.glob(
        os.path.join(
            base_dir, f'**/*{exp_id}*/**/collect/**/*events.out.tfevents*'
        ),
        recursive=True,
    )
    event_log_dirs.extend(logs)

  event_log_dirs = set(event_log_dirs)
  logging.info(f'Found {len(event_log_dirs)} event log dirs')

  process_log_dir_fn = functools.partial(
      process_tb_event_dir, tags=event_file_tags
  )

  all_dfs = []
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for data in executor.map(process_log_dir_fn, event_log_dirs):
      if data is not None:
        # print logging message based on the log dir
        print(
            'Processing log dir:'
            f' {data.iloc[0]["domain"]}/{data.iloc[0]["task"]}/{data.iloc[0]["algo"]}/{data.iloc[0]["experiment"]}/{data.iloc[0]["seed"]}'
        )
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
  logging.info(f'Row counts after removing duplicate steps: {row_counts}')

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
  csv_files = []
  for exp_id in experiment_ids:
    logs = glob.glob(
        os.path.join(base_dir, f'**/*{exp_id}*/**/*train_emissions.csv'),
        recursive=True,
    )
    csv_files.extend(logs)

  csv_files = set(csv_files)
  logging.info(f'Found {len(csv_files)} csv files')

  process_codecarbon_csv_fn = functools.partial(process_codecarbon_csv)

  all_dfs = []
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for data in executor.map(process_codecarbon_csv_fn, csv_files):
      if data is not None:
        all_dfs.append(data)
  metrics_df = pd.concat(all_dfs)
  logging.info(f'Loaded {len(metrics_df)} rows of data')

  return metrics_df


def load_inference_reward_data(base_dir, experiment_ids):
  pass


def load_inference_system_data(base_dir, experiment_ids):
  pass


def main(_):
  tf.compat.v1.enable_eager_execution()

  np.random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)
  _initialize_plotting()

  training_reward_data_df = load_training_reward_data(
      base_dir=_BASE_DIR.value, experiment_ids=_EXPERIMENT_IDS.value
  )
  print(training_reward_data_df.head())
  training_reward_metrics = analysis.reliability.get_training_metrics(
      data_df=training_reward_data_df, tag='Metrics/AverageReturn', index='Step'
  )

  training_system_metrics_df = load_training_system_data(
      base_dir=_BASE_DIR.value, experiment_ids=_EXPERIMENT_IDS.value
  )
  print(training_system_metrics_df.head())
  training_system_metrics = analysis.system.get_training_metrics(
      data_df=training_system_metrics_df
  )

  # Display all training metrics
  training_metrics = dict(**training_reward_metrics, **training_system_metrics)
  training_metrics_df = analysis.results.metrics_dict_to_pandas_df(
      training_metrics
  )
  # Print out the latex table with training metrics only
  print(analysis.results.df_as_latex(training_metrics_df, mode='train'))
  #
  # inference_reward_metrics_df = load_inference_reward_data(
  #     base_dir=_BASE_DIR.value,
  #     experiment_ids=_EXPERIMENT_IDS.value)
  # print(inference_reward_metrics_df.head())
  # inference_reward_metrics = get_inference_reward_metrics(
  #     data_df=inference_reward_metrics_df)
  #
  # inference_system_metrics_df = load_inference_system_data(
  #     base_dir=_BASE_DIR.value,
  #     experiment_ids=_EXPERIMENT_IDS.value)
  # print(inference_system_metrics_df.head())
  # inference_system_metrics = get_inference_system_metrics(
  #     data_df=inference_system_metrics_df)


if __name__ == '__main__':
  app.run(main)
