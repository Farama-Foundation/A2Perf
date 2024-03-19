import collections
import logging

import numpy as np
import pandas as pd
import scipy

LEFT_TAIL_ALPHA = 0.05
RIGHT_TAIL_ALPHA = 0.95

MIN_NUM_DISPERSION_DATA_POINTS = 2


def compute_dispersion(curve):
  # Compute the dispersion as the IQR of the curve
  return scipy.stats.iqr(curve)


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


def lowpass_filter(curve, lowpass_thresh):
  filt_b, filt_a = scipy.signal.butter(8, lowpass_thresh)

  def butter_filter_fn(c):
    padlen = min(len(c) - 1, 3 * max(len(filt_a), len(filt_b)))
    return scipy.signal.filtfilt(filt_b, filt_a, curve, padlen=padlen)

  processed_curve = butter_filter_fn(curve)
  return processed_curve


def apply_lowpass(curve):
  # Apply the lowpass filter directly on the curve
  low_pass_curve = lowpass_filter(curve, lowpass_thresh=0.01)
  return low_pass_curve


def dispersion_across_runs(data_df, tag, index):
  step_col = f'{tag}_{index}'
  value_col = f'{tag}_Value'
  lowpass_value_col = f'{tag}_lowpass_Value'

  # Lowpass filter each curve
  data_df[lowpass_value_col] = data_df.groupby(
      ['domain', 'task', 'algo', 'experiment', 'seed']
  )[value_col].transform(apply_lowpass)

  # Group the curves by 'domain', 'algo', 'task', and 'step_col' to compute dispersion
  dispersion_groups = data_df.groupby(['domain', 'algo', 'task', step_col])

  # Log all groups just to make sure we're capturing correct data
  logging.info('Groups: %s', dispersion_groups)

  def compute_dispersion_if_enough_data(group):
    if len(group) > MIN_NUM_DISPERSION_DATA_POINTS:
      return compute_dispersion(group)
    else:
      logging.warning('Insufficient data at step %s for tag %s. Skipping'
                      ' dispersion calculation.', group.name[-1], tag)
      return None  # or return some default value

  dispersion_df = (
      dispersion_groups[lowpass_value_col]
      .apply(compute_dispersion_if_enough_data)
      .reset_index()
  )

  # Drop rows where '{tag}_lowpass_Value' column has NaN values
  lowpass_value_col = f'{tag}_lowpass_Value'
  dispersion_df = dispersion_df.dropna(subset=[lowpass_value_col])

  # Renaming the column for clarity
  dispersion_df.rename(
      columns={lowpass_value_col: 'iqr_dispersion'}, inplace=True
  )

  metrics = {}
  for (domain, algo, task), group in dispersion_df.groupby(
      ['domain', 'algo', 'task']
  ):
    mean_iqr = group['iqr_dispersion'].mean()
    std_iqr = group['iqr_dispersion'].std()
    metrics[(
        domain,
        algo,
        task,
    )] = dict(mean=mean_iqr, std=std_iqr)
    logging.info('Domain: %s, Task: %s, Algo: %s, Mean IQR: %s, Std IQR: %s',
                 domain, task, algo, mean_iqr, std_iqr)
  return metrics


def compute_eval_points_within_runs(
    data_df, tag, index, eval_points_per_window=5
):
  experiment_meta_data = {}
  for (domain, task, algo, experiment, seed), group in data_df.groupby(
      ['domain', 'task', 'algo', 'experiment', 'seed']
  ):
    df = group.sort_values(by=f'{tag}_{index}').copy()
    median_step_diff = df[f'{tag}_{index}'].diff().median()
    window_size = int(eval_points_per_window * median_step_diff)
    eval_points = list(
        range(
            np.ceil(window_size / 2).astype(int),
            max(df[f'{tag}_{index}']),
            int(median_step_diff),
        )
    )

    logging.info(
        'Domain: %s, Algo: %s, Experiment: %s, Task: %s, Seed: %s',
        domain, algo, experiment, task, seed
    )
    logging.info('\tMedian step difference: %s', median_step_diff)
    logging.info('\tWindow size: %s', window_size)
    logging.info('\tNum eval points: %s', len(eval_points))

    experiment_meta_data[(domain, task, algo, experiment, seed)] = {
        'eval_points': eval_points,
        'window_size': window_size,
        'median_step_diff': median_step_diff,
    }

  return experiment_meta_data


def dispersion_within_runs(
    data_df,
    tag,
    index,
    experiment_meta_data,
    dispersion_window_fn=scipy.stats.iqr,
):
  metrics = {}
  for (domain, task, algo), group in data_df.groupby(
      ['domain', 'task', 'algo']
  ):
    print(f'Processing Domain: {domain}, Task: {task}, Algo: {algo}')
    all_iqr_values = []
    for exp_id, exp_group in group.groupby('experiment'):
      for seed, seed_group in exp_group.groupby('seed'):
        seed_group[f'{tag}_Value_diff'] = seed_group[f'{tag}_Value'].diff()
        seed_group = seed_group[[f'{tag}_{index}', f'{tag}_Value_diff']].copy()
        seed_group = seed_group.dropna()
        steps, episode_reward_diff = seed_group.to_numpy().T

        window_size = experiment_meta_data[(domain, task, algo, exp_id, seed)][
          'window_size'
        ]
        for eval_point in experiment_meta_data[
          (domain, task, algo, exp_id, seed)
        ]['eval_points']:
          low_end = np.ceil(eval_point - (window_size / 2))
          high_end = np.floor(eval_point + (window_size / 2))

          eval_points_above = steps >= low_end
          eval_points_below = steps <= high_end
          eval_points_in_window = np.logical_and(
              eval_points_above, eval_points_below
          )
          valid_eval_points = np.nonzero(eval_points_in_window)[0]

          if len(valid_eval_points) == 0:
            logging.warning(
                'No valid eval points for domain: %s, task: %s, algo: %s,'
                ' exp_id: %s, seed: %s, eval_point: %s',
                domain, task, algo, exp_id, seed, eval_point)

            continue
          elif len(valid_eval_points) < 2:
            # IQR needs at least 2 data points for meaningful calculation
            logging.warning(
                'Insufficient data points for IQR calculation for domain: %s,'
                ' task: %s, algo: %s, exp_id: %s, seed: %s, eval_point: %s',
                domain, task, algo, exp_id, seed, eval_point
            )
            continue

          # Apply window_fn to get the IQR for the current window
          window_dispersion = dispersion_window_fn(
              episode_reward_diff[valid_eval_points]
          )
          all_iqr_values.append(window_dispersion)
    mean_iqr = np.mean(all_iqr_values)
    std_iqr = np.std(all_iqr_values)
    metrics[(
        domain,
        algo,
        task,
    )] = dict(mean=mean_iqr, std=std_iqr)
    print(
        f'Domain: {domain}, Task: {task}, Algo: {algo}, Mean IQR: {mean_iqr},'
        f' Std IQR: {std_iqr}'
    )

  return metrics


def short_term_risk(data_df, tag, index):
  metrics = {}
  for (domain, task, algo), group in data_df.groupby(
      ['domain', 'task', 'algo']
  ):
    logging.info('Processing Domain: %s, Task: %s, Algo: %s', domain, task,
                 algo)

    all_diffs = []
    for exp_id, exp_group in group.groupby('experiment'):
      for seed, seed_group in exp_group.groupby('seed'):
        seed_df = seed_group.sort_values(by=f'{tag}_{index}').copy()
        seed_df[f'{tag}_Value_diff'] = seed_df[f'{tag}_Value'].diff()
        seed_df = seed_df.dropna()

        episode_reward_diffs = seed_df[f'{tag}_Value_diff'].values
        all_diffs.extend(episode_reward_diffs)

    risk = np.percentile(all_diffs, LEFT_TAIL_ALPHA * 100, method='linear')

    # CVaR is the average of the bottom "alpha" percent of diffs
    all_diffs = np.array(all_diffs)
    cvar = np.mean(all_diffs[all_diffs <= risk])
    cvar = -cvar  # make it positive for easier interpretation
    logging.info('\t\t\tCVaR: %s', cvar)
    metrics[(
        domain,
        algo,
        task,
    )] = cvar
  return metrics


def long_term_risk(data_df, tag, index):
  """Calculate the Conditional Value at Risk (CVaR) for different experimental groups.

  Args:

  data_df (DataFrame): The dataset containing the experimental results.
  tag (str): The tag used to identify the relevant columns in data_df.
  index (str): The index used to sort the data within each group.

  Returns:
  dict: A dictionary containing the CVaR for each (domain, algorithm, task)
  combination.
  """

  metrics = {}
  for (domain, task, algo), group in data_df.groupby(
      ['domain', 'task', 'algo']
  ):
    logging.info('Processing Domain: %s, Task: %s, Algo: %s', domain, task,
                 algo)
    all_drawdowns = []

    for exp_id, exp_group in group.groupby('experiment'):
      for seed, seed_group in exp_group.groupby('seed'):
        # Ensure the correct column exists
        if f'{tag}_{index}' not in seed_group.columns:
          logging.warning('Column %s not found in data.', f'{tag}_{index}')
          continue

        # Sort the seed group by the index
        seed_group = seed_group.sort_values(by=f'{tag}_{index}')

        # Compute the drawdowns
        values = seed_group[f'{tag}_Value'].values
        drawdowns = compute_drawdown(values)
        all_drawdowns.extend(drawdowns)

    if all_drawdowns:
      # Get the worst alpha percentile of drawdowns
      top_alpha_percent = np.percentile(all_drawdowns, RIGHT_TAIL_ALPHA * 100)
      all_drawdowns = np.array(all_drawdowns)

      # CVaR is the average of the worst "alpha" percent of drawdowns
      cvar = np.mean(all_drawdowns[all_drawdowns >= top_alpha_percent])
      logging.info('\t\t\tCVaR: %s', cvar)
      metrics[(domain, algo, task)] = cvar
    else:
      logging.warning('No drawdown data available for %s, %s, %s', domain, task,
                      algo)
  return metrics


def risk_across_runs(data_df, tag, alpha=0.05):
  """Calculate the Conditional Value at Risk (CVaR) for the final values across different runs.

  Args:

  data_df (DataFrame): The dataset containing the experimental results.
  tag (str): The tag used to identify the relevant value columns in data_df.
  index (str): The index used to sort the data within each group.
  alpha (float): The percentile for CVaR calculation (default is 0.05).

  Returns:
  DataFrame: A DataFrame with CVaR for each (domain, task, algo) combination.
  """

  metrics = {}

  # Extract the final values for each group
  final_values_col = f'{tag}_Value'
  final_tag_values = (
      data_df.groupby(['domain', 'task', 'algo', 'experiment', 'seed'])[
        final_values_col
      ]
      .last()
      .reset_index()
  )

  # Grouping all experiments and seeds within a specific domain/task/algo
  for (domain, task, algo), group in final_tag_values.groupby(
      ['domain', 'task', 'algo']
  ):
    # Get the bottom "alpha" percentile of final values
    values = group[final_values_col].values
    bottom_alpha_percent = np.percentile(values, alpha * 100, method='linear')
    cvar = np.mean(values[values <= bottom_alpha_percent])

    # Logging the process and results
    logging.info('Processing Domain: %s, Task: %s, Algo: %s', domain, task
                 , algo)
    logging.info('\t\t\tCVaR: %s', cvar)
    # Storing the CVaR values in the metrics dictionary
    metrics[(domain, algo, task)] = cvar

  return metrics


def get_training_metrics(data_df, tag, index):
  dispersion_across_runs_result = dispersion_across_runs(data_df, tag, index)
  experiment_meta_data = compute_eval_points_within_runs(
      data_df=data_df, tag=tag, index=index
  )
  dispersion_within_runs_result = dispersion_within_runs(
      data_df=data_df,
      tag=tag,
      index=index,
      experiment_meta_data=experiment_meta_data,
  )
  short_term_risk_result = short_term_risk(
      data_df=data_df, tag=tag, index=index
  )
  long_term_risk_result = long_term_risk(data_df=data_df, tag=tag, index=index)
  risk_across_runs_result = risk_across_runs(
      data_df=data_df, tag=tag, alpha=LEFT_TAIL_ALPHA
  )

  return {
      'dispersion_across_runs': dispersion_across_runs_result,
      'dispersion_within_runs': dispersion_within_runs_result,
      'short_term_risk': short_term_risk_result,
      'long_term_risk': long_term_risk_result,
      'risk_across_runs': risk_across_runs_result,
  }


def dispersion_across_rollouts(data_df):
  metrics = {}
  for (domain, algo, task), group in data_df.groupby(
      ['domain', 'algo', 'task']):

    # We are only interested in the `rollout_returns` metric
    group = group[group['metric'] == 'rollout_returns']
    all_values = []
    for values in group['values']:
      all_values.extend(values)

    all_values = np.array(all_values)

    iqr = scipy.stats.iqr(all_values)
    logging.info('Processing Domain: %s, Task: %s, Algo: %s', domain, task,
                 algo)
    logging.info('\t\t\tIQR: %s', iqr)

    metrics[(domain, algo, task)] = iqr
  return metrics


def risk_across_rollouts(data_df):
  metrics = {}
  for (domain, algo, task), group in data_df.groupby(
      ['domain', 'algo', 'task']):

    # We are only interested in the `rollout_returns` metric
    group = group[group['metric'] == 'rollout_returns']
    all_values = []
    for values in group['values']:
      all_values.extend(values)

    all_values = np.array(all_values)
    bottom_alpha_percent = np.percentile(all_values, LEFT_TAIL_ALPHA * 100,
                                         method='linear')
    cvar = np.mean(all_values[all_values <= bottom_alpha_percent])
    logging.info('Processing Domain: %s, Task: %s, Algo: %s', domain, task,
                 algo)
    logging.info('\t\t\tCVaR: %s', cvar)

    metrics[(domain, algo, task)] = cvar
  return metrics


def get_inference_metrics(data_df):
  risk_across_rollouts_result = risk_across_rollouts(data_df)
  dispersion_across_rollouts_result = dispersion_across_rollouts(data_df)
  return {
      'risk_across_rollouts': risk_across_rollouts_result,
      'dispersion_across_rollouts': dispersion_across_rollouts_result,
  }
