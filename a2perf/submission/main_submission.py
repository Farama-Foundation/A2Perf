import multiprocessing
import os

import gin
from absl import app
from absl import flags

from a2perf.submission.submission_util import Submission

_GIN_CONFIG = flags.DEFINE_string('gin_config', None,
                                  'Path to the gin-config file.')
_PARTICIPANT_MODULE_PATH = flags.DEFINE_string(
    'participant_module_path', None, 'Path to participant module.'
)
_ROOT_DIR = flags.DEFINE_string(
    'root_dir', '/tmp/xm_local', 'Base directory for logs and results.'
)
_METRIC_VALUES_DIR = flags.DEFINE_string(
    'metric_values_dir', None, 'Directory to save metrics values.'
)
_TRAIN_LOGS_DIRS = flags.DEFINE_multi_string(
    'train_logs_dirs',
    ['train_logs'],
    'Directories for train logs from all of the experiments that reliability'
    ' metrics will be calculated on',
)
_EXTRA_GIN_BINDINGS = flags.DEFINE_multi_string(
    'extra_gin_bindings',
    [],
    'Extra gin bindings to add configurations on the fly.',
)
_RUN_OFFLINE_METRICS_ONLY = flags.DEFINE_bool(
    'run_offline_metrics_only', False, 'Whether to run offline metrics only.'
)


def main(_):
  # Set the working directory to the submission directory.
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  multiprocessing.set_start_method('spawn', force=False)

  print('gin_config:', _GIN_CONFIG.value)
  print('participant_module_path:', _PARTICIPANT_MODULE_PATH.value)

  gin.parse_config_file(_GIN_CONFIG.value)
  for binding in _EXTRA_GIN_BINDINGS.value:
    gin.parse_config(binding)
    print(binding)

  submission = Submission(
      root_dir=_ROOT_DIR.value,
      metric_values_dir=_METRIC_VALUES_DIR.value,
      participant_module_path=_PARTICIPANT_MODULE_PATH.value,
      train_logs_dirs=_TRAIN_LOGS_DIRS.value,
      run_offline_metrics_only=_RUN_OFFLINE_METRICS_ONLY.value,
  )
  submission.run_benchmark()

  # multiprocessing make sure all processes are terminated
  for p in multiprocessing.active_children():
    p.terminate()
    p.join()


if __name__ == '__main__':
  app.run(main)
