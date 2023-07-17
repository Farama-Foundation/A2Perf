import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
import gin
from rl_perf.submission.submission_util import Submission

flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
flags.DEFINE_string(
    'participant_module_path', None, 'Path to participant module.'
)
flags.DEFINE_string(
    'root_dir', '/tmp/xm_local', 'Base directory for logs and results.'
)
flags.DEFINE_string(
    'metric_values_dir', None, 'Directory to save metrics values.'
)
flags.DEFINE_list(
    'train_logs_dirs',
    ['train_logs'],
    'Directories for train logs from all of the experiments that reliability'
    ' metrics will be calculated on',
)

flags.DEFINE_bool(
    'run_offline_metrics_only', False, 'Whether to run offline metrics only.'
)
FLAGS = flags.FLAGS


def main(_):
  # Set the working directory to the submission directory.
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  multiprocessing.set_start_method('spawn', force=True)

  print('FLAGS.gin_file', FLAGS.gin_file)
  print('FLAGS.participant_module_path', FLAGS.participant_module_path)

  gin.parse_config_file(FLAGS.gin_file)
  logging.set_verbosity(logging.INFO)

  submission = Submission(
      root_dir=FLAGS.root_dir,
      metric_values_dir=FLAGS.metric_values_dir,
      participant_module_path=FLAGS.participant_module_path,
      train_logs_dirs=FLAGS.train_logs_dirs,
      run_offline_metrics_only=FLAGS.run_offline_metrics_only,
  )
  submission.run_benchmark()


if __name__ == '__main__':
  app.run(main)
