from functools import reduce

import pandas as pd


def get_distributed_experiment_metric(data_df, metric,
    tolerance=pd.Timedelta('10sec'), dtype=float):
  data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
  final_dfs_to_concat = []

  for _, group in data_df.groupby(
      ['domain', 'algo', 'task', 'experiment', 'seed']):
    group = group.sort_values('timestamp')

    # Initialize list to hold data frames for each run_id
    run_groups = []
    run_ids = group['run_id'].unique()

    # Create a separate DataFrame for each run_id with the specific metric column renamed
    for run_id in run_ids:
      run_group = group[group['run_id'] == run_id][
        ['timestamp', 'domain', 'algo', 'task', 'experiment', 'seed',
         metric]].rename(
          columns={metric: f'{metric}_{run_id}'}
      ).astype({f'{metric}_{run_id}': dtype})
      run_groups.append(run_group)

    # Use functools.reduce to merge all DataFrame objects at once
    merged_group = reduce(
        lambda left, right: pd.merge_asof(
            left, right, on='timestamp', tolerance=tolerance,
            suffixes=('', f'_{right.columns[-1]}')
        ), run_groups
    )

    metric_columns = [col for col in merged_group.columns if
                      col.startswith(metric)]
    merged_group[f'experiment_{metric}'] = merged_group[metric_columns].sum(
        axis=1)
    final_dfs_to_concat.append(merged_group[
                                 ['domain', 'algo', 'task', 'experiment',
                                  'seed', 'timestamp', f'experiment_{metric}']])

  # Concatenate all final DataFrames
  aggregated_df = pd.concat(final_dfs_to_concat, ignore_index=True)
  return aggregated_df


def get_metric(data_df, metric, tolerance=pd.Timedelta('10sec')):
  metric_df = get_distributed_experiment_metric(data_df, metric, tolerance)
  experiment_metric_col_name = f'experiment_{metric}'

  metrics = {}
  for (domain, algo, task), group in metric_df.groupby(
      ['domain', 'algo', 'task']
  ):
    mean_metric = group[experiment_metric_col_name].mean()
    std_metric = group[experiment_metric_col_name].std()
    metrics[(domain, algo, task)] = {'mean': mean_metric, 'std': std_metric}
  return metrics


def get_mean_ram_usage(data_df, tolerance=pd.Timedelta('10sec')):
  ram_usage_df = get_distributed_experiment_metric(
      data_df, 'ram_process', tolerance=tolerance
  )
  metrics = {}

  for (domain, algo, task), group in ram_usage_df.groupby(
      ['domain', 'algo', 'task']
  ):
    mean_ram_usage = group['experiment_ram_process'].mean()
    std_ram_usage = group['experiment_ram_process'].std()

    metrics[(domain, algo, task)] = {
        'mean': mean_ram_usage,
        'std': std_ram_usage,
    }

  return metrics


def get_gpu_power_usage(data_df):
  gpu_power_usage_df = get_distributed_experiment_metric(data_df, 'gpu_power')
  metrics = {}
  for (domain, algo, task), group in gpu_power_usage_df.groupby(
      ['domain', 'algo', 'task']
  ):
    mean_gpu_power_usage = group['experiment_gpu_power'].mean()
    std_gpu_power_usage = group['experiment_gpu_power'].std()

    metrics[(domain, algo, task)] = {
        'mean': mean_gpu_power_usage,
        'std': std_gpu_power_usage,
    }

  return metrics


def get_peak_ram_usage(data_df, tolerance=pd.Timedelta('10sec')):
  ram_usage_df = get_distributed_experiment_metric(
      data_df, 'ram_process', tolerance=tolerance
  )
  # Find the max ram usage for each experiment
  peak_ram_usage = ram_usage_df.groupby(
      ['domain', 'algo', 'task', 'experiment', 'seed']
  )['experiment_ram_process'].max()
  metrics = {}
  for (domain, algo, task), group in peak_ram_usage.groupby(
      ['domain', 'algo', 'task']
  ):
    metrics[(domain, algo, task)] = {'mean': group.mean(), 'std': group.std()}

  return metrics


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
      ['domain', 'algo', 'task', 'experiment', 'seed']
  ).max()
  latest_shared_timestamp = latest_timestamp.groupby(
      ['domain', 'algo', 'task', 'experiment', 'seed']
  ).min()

  # Compute wall clock time in hours
  wall_clock_time = (
                        latest_shared_timestamp - earliest_shared_timestamp
                    ).dt.total_seconds() / 3600

  # Group by 'domain', 'algo', 'task' and calculate mean and std of wall clock time
  metrics = {}
  grouped_wall_clock = wall_clock_time.groupby(['domain', 'algo', 'task'])
  for (domain, algo, task), group_data in grouped_wall_clock:
    metrics[(domain, algo, task)] = {
        'mean': group_data.mean(),
        'std': group_data.std(),
    }

  return metrics


def get_ram_power_usage(data_df):
  # Not implemented since this is an estimate in codecarbon
  return {}


def get_cpu_power_usage(data_df):
  # Not implemented since this is an estimate without access to Intel RAPL
  return {}


def get_total_energy(data_df):
  energy_consumed_all = data_df.groupby(
      ['domain', 'algo', 'task', 'experiment', 'seed', 'run_id']
  )['energy_consumed'].last().astype(float)

  total_energy_consumed = energy_consumed_all.groupby(
      ['domain', 'algo', 'task', 'experiment', 'seed']
  ).sum()

  # Group by 'domain', 'algo', 'task' and calculate mean and std of total energy
  metrics = {}
  grouped_total_energy = total_energy_consumed.groupby(
      ['domain', 'algo', 'task'])
  for (domain, algo, task), group_data in grouped_total_energy:
    metrics[(domain, algo, task)] = {
        'mean': group_data.mean(),
        'std': group_data.std(),
    }

  return metrics


def get_power_usage(data_df):
  ram_power_usage = get_ram_power_usage(data_df)
  gpu_power_usage = get_gpu_power_usage(data_df)
  cpu_power_usage = get_cpu_power_usage(data_df)

  return {
      'gpu_power_usage': gpu_power_usage,
  }


def get_training_metrics(data_df):
  wall_clock_time = get_wall_clock_time(data_df=data_df)
  mean_ram_usage = get_mean_ram_usage(data_df=data_df)
  peak_ram_usage = get_peak_ram_usage(data_df=data_df)
  power_usage = get_power_usage(data_df=data_df)
  energy_consumed = get_total_energy(data_df=data_df)
  return {
      'mean_ram_usage': mean_ram_usage,
      'peak_ram_usage': peak_ram_usage,
      'wall_clock_time': wall_clock_time,
      'energy_consumed': energy_consumed,
      **power_usage,
  }


def get_inference_metrics(data_df):
  mean_ram_usage = get_mean_ram_usage(data_df=data_df)
  peak_ram_usage = get_peak_ram_usage(data_df=data_df)
  power_usage = get_power_usage(data_df=data_df)

  return {
      'mean_ram_usage': mean_ram_usage,
      'peak_ram_usage': peak_ram_usage,
      **power_usage,
  }
