import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from absl import app
from absl import flags

from a2perf import analysis
from a2perf.analysis.metrics_lib import load_inference_metric_data
from a2perf.analysis.metrics_lib import load_inference_system_data
from a2perf.analysis.metrics_lib import load_training_reward_data
from a2perf.analysis.metrics_lib import load_training_system_data
from a2perf.analysis.metrics_lib import plot_training_reward_data

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
  base_font_size = 14
  sns.set_style('whitegrid')
  plt.rcParams.update({
      'figure.figsize': (12, 6),
      'font.size': base_font_size - 2,
      'axes.labelsize': base_font_size - 2,
      'axes.titlesize': base_font_size,
      'axes.labelweight': 'bold',  # Bold font for the axes labels
      'legend.fontsize': base_font_size - 4,
      'xtick.labelsize': base_font_size - 4,
      'ytick.labelsize': base_font_size - 4,
      'figure.titlesize': base_font_size,
      'figure.dpi': 100,
      'savefig.dpi': 100,
      'savefig.format': 'png',
      'savefig.bbox': 'tight',
      'grid.linewidth': 0.5,
      'grid.alpha': 0.5  # Lighter grid lines
  })


def main(_):
  tf.compat.v1.enable_eager_execution()

  np.random.seed(_SEED.value)
  tf.random.set_seed(_SEED.value)
  _initialize_plotting()

  base_dir = os.path.expanduser(_BASE_DIR.value)
  training_reward_data_df = load_training_reward_data(
      base_dir=base_dir, experiment_ids=_EXPERIMENT_IDS.value
  )

  plot_training_reward_data(training_reward_data_df,
                            event_file_tags=['Metrics/AverageReturn'])

  training_reward_metrics = analysis.reliability.get_training_metrics(
      data_df=training_reward_data_df, tag='Metrics/AverageReturn', index='Step'
  )
  training_system_metrics_df = load_training_system_data(
      base_dir=base_dir, experiment_ids=_EXPERIMENT_IDS.value
  )
  training_system_metrics = analysis.system.get_training_metrics(
      data_df=training_system_metrics_df
  )
  training_metrics = dict(**training_reward_metrics, **training_system_metrics)

  inference_reward_metrics, inference_reward_metrics_df = load_inference_metric_data(
      base_dir=base_dir, experiment_ids=_EXPERIMENT_IDS.value
  )
  inference_reward_metrics.update(analysis.reliability.get_inference_metrics(
      data_df=inference_reward_metrics_df))
  inference_system_metrics_df = load_inference_system_data(
      base_dir=base_dir,
      experiment_ids=_EXPERIMENT_IDS.value)
  inference_system_metrics = analysis.system.get_inference_metrics(
      data_df=inference_system_metrics_df)
  inference_metrics = dict(**inference_reward_metrics,
                           **inference_system_metrics)

  # Take rollout_returns from inference_metrics and add it to training_metrics
  training_metrics['rollout_returns'] = inference_metrics['rollout_returns']
  del inference_metrics['rollout_returns']

  training_metrics_df = analysis.results.metrics_dict_to_pandas_df(
      training_metrics
  )
  inference_metrics_df = analysis.results.metrics_dict_to_pandas_df(
      inference_metrics
  )

  print(analysis.results.df_as_latex(training_metrics_df, mode='train'))
  print(analysis.results.df_as_latex(inference_metrics_df, mode='inference'))


if __name__ == '__main__':
  app.run(main)
