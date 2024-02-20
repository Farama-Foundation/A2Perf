import collections

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

  return None


def dispersion_within_runs(data_df, tag, index):
  pass


def short_term_risk(data_df, tag, index):
  pass


def long_term_risk(data_df, tag, index):
  pass


def risk_across_runs(data_df, tag, index):
  pass


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
