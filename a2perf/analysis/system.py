import pandas as pd


def get_inference_time():
  # Mean/STD of the inference time metric
  inference_time_df = inference_df[inference_df['metric'] == 'inference_time']
  mean_inference_time = inference_time_df.groupby(['domain', 'algo', 'task', ])[
    0].mean()
  std_inference_time = inference_time_df.groupby(['domain', 'algo', 'task', ])[
    0].std()
  mean_inference_time, std_inference_time

  pass


def get_peak_ram_usage(data_df):
  training_peak_ram_usage = \
    training_system_metrics_df.groupby(
        ['domain', 'algo', 'task', 'experiment', 'seed'])[
      'ram_process'].max()

  training_mean_peak_ram = training_peak_ram_usage.groupby(
      ['domain', 'algo', 'task']).mean()
  training_std_peak_ram = training_peak_ram_usage.groupby(
      ['domain', 'algo', 'task']).std()


def get_mean_ram_usage(data_df):
  # Get the mean RAM used for
  ram_usage = \
  data_df.groupby(['domain', 'algo', 'task', 'experiment', 'seed', 'run_id'])[
    'ram_process']



def get_wall_clock_time(data_df):
  # Ensure timestamps are in datetime format
  if not pd.api.types.is_datetime64_any_dtype(data_df['timestamp']):
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])

  # Find the earliest and latest timestamps for each run
  group_cols = ['domain', 'algo', 'task', 'experiment', 'seed', 'run_id']
  earliest_timestamp = data_df.groupby(group_cols)['timestamp'].min()
  latest_timestamp = data_df.groupby(group_cols)['timestamp'].max()

  # Find the earliest shared and latest shared timestamps for each experiment
  earliest_shared_timestamp = earliest_timestamp.groupby(
      ['domain', 'algo', 'task', 'experiment', 'seed']).max()
  latest_shared_timestamp = latest_timestamp.groupby(
      ['domain', 'algo', 'task', 'experiment', 'seed']).min()

  # Compute wall clock time in seconds
  wall_clock_time = (
      latest_shared_timestamp - earliest_shared_timestamp).dt.total_seconds()

  # Group by 'domain', 'algo', 'task' and calculate mean and std of wall clock time
  metrics = {}
  grouped_wall_clock = wall_clock_time.groupby(['domain', 'algo', 'task'])
  for (domain, algo, task), group_data in grouped_wall_clock:
    metrics[(domain, algo, task)] = {
        'mean': group_data.mean(),
        'std': group_data.std()
    }

  return metrics


def get_power_usage(data_df):
  pass


def get_energy_consumed(data_df):
  pass


def get_training_metrics(data_df):
  wall_clock_time = get_wall_clock_time(data_df=data_df)
  mean_ram_usage = get_mean_ram_usage(data_df=data_df)
  peak_ram_usage = get_peak_ram_usage(data_df=data_df)
  power_usage = get_power_usage(data_df=data_df)
  energy_consumed = get_energy_consumed(data_df=data_df)

  return {
      'mean_ram_usage': mean_ram_usage,
      'peak_ram_usage': peak_ram_usage,
      'power_usage': power_usage,
      'energy_consumed': energy_consumed,
      'wall_clock_time': wall_clock_time
  }
