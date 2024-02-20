import collections
import logging

import numpy as np
import pandas as pd

import scipy


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


def dispersion_across_rollouts(data_df, tag, index):
  return risk_across_rollouts(data_df, tag, index)


def risk_across_rollouts(data_df, tag, index):
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


def dispersion_across_runs(data_df, tag, index):
  # Create useful indices for Pandas
  tag_step_index = pd.MultiIndex.from_tuples([(tag, index)])
  tag_value_index = pd.MultiIndex.from_tuples([(tag, 'Value')])
  lowpass_value_index = pd.MultiIndex.from_tuples(
      [(f'{tag}_lowpass', index)])

  # Lowpass filter each curve
  data_df[lowpass_value_index] = \
    data_df.groupby(['domain', 'task', 'algo', 'experiment', 'seed'])[
      tag_value_index].transform(apply_lowpass)

  # Group the curves by index to compute dispersion
  dispersion_groups = data_df.groupby(
      ['domain', 'algo', 'task', tag_step_index])
  dispersion_df = dispersion_groups[lowpass_value_index].apply(
      compute_dispersion).reset_index()

  # # Renaming the column for clarity
  # dispersion_df.rename(columns={'lowpass_episode_reward': 'iqr_dispersion'},
  #                      inplace=True)
  # # Now that we have the dispersion values for specific steps, we can compute the mean and std of the dispersion for each algo
  # mean_dispersion = dispersion_df.groupby(['domain', 'algo', 'task'])[
  #   'iqr_dispersion'].mean()
  # std_dispersion = dispersion_df.groupby(['domain', 'algo', 'task'])[
  #   'iqr_dispersion'].std()
  #
  # # In[ ]:
  #
  # # Display the means and stds nicely. Should be durations for domain/algo/task
  # for domain in mean_dispersion.index.levels[0]:
  #   print(domain)
  #   for algo in mean_dispersion.index.levels[1]:
  #     print(f'\t{algo}')
  #     for task in mean_dispersion.loc[domain, algo].index:
  #       print(
  #           f'\t\tTask: {task}, Mean Dispersion: {mean_dispersion.loc[domain, algo, task]}, Std Dispersion: {std_dispersion.loc[domain, algo, task]}')
  #       all_training_metrics[(domain, algo, task)][
  #         'dispersion_across_runs'] = dict(
  #           mean=mean_dispersion.loc[domain, algo, task],
  #           std=std_dispersion.loc[domain, algo, task])

  return dispersion_df


def dispersion_within_runs(data_df, tag, index):
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


def short_term_risk(data_df, tag, index):
  alpha = 0.05

  for domain, domain_group in data_df.groupby('domain'):
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


def long_term_risk(data_df, tag, index):
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


def risk_across_runs(data_df, tag, index):
  tag_value_index = pd.MultiIndex.from_tuples([(tag, 'Value')])
  final_tag_value = data_df.groupby(['domain', 'task', 'algo', 'seed'])[
    tag_value_index].last().reset_index()

  # Now compute the CVaR for each domain/task/algo
  alpha = 0.05

  risk_df = collections.defaultdict(dict)

  # all experiments and seeds within a specific domain/task/algo should be grouped together
  for domain, domain_group in final_tag_value.groupby('domain'):
    for task, task_group in domain_group.groupby('task'):
      for algo, algo_group in task_group.groupby('algo'):
        # Get the bottom "alpha" percent of final episode rewards
        rewards = algo_group['episode_reward'].values
        bottom_alpha_percent = np.percentile(rewards, alpha * 100,
                                             method='linear')
        cvar = np.mean(rewards[rewards <= bottom_alpha_percent])

        logging.info(f'Processing domain: {domain}, task: {task}, algo: {algo}')
        logging.info(f'\tCVaR: {cvar}')

        # Create a small df to hold the final values

        risk_df[(domain, algo, task,)] = {
            'risk_across_runs': cvar
        }

  return pd.DataFrame.from_dict(risk_df, orient='index')


def get_training_reliability_metrics(data_df, tag, index):
  dispersion_across_runs_result = dispersion_across_runs(data_df, tag, index)
  dispersion_within_runs_result = dispersion_within_runs(data_df, tag, index)
  short_term_risk_result = short_term_risk(data_df, tag, index)
  long_term_risk_result = long_term_risk(data_df, tag, index)
  risk_across_runs_result = risk_across_runs(data_df, tag, index)

  return None


def get_inference_reliability_metrics(data_df):
  # Dispersion Across Rollouts (IQR)
  # Risk Across Rollouts (CVaR)
  return None
