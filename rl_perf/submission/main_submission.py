import multiprocessing
import os

import gin
from absl import app
from absl import flags
from absl import logging
from rl_perf.submission.submission_util import Submission

flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
flags.DEFINE_string('participant_module_path', None, 'Path to participant module.')
flags.DEFINE_string('base_log_dir', '/tmp/xm_local', 'Base directory for logs and results.')
flags.DEFINE_string('metric_values_dir', None, 'Directory to save metrics values.')
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
        base_log_dir=FLAGS.base_log_dir,
        metric_values_dir=FLAGS.metric_values_dir,
        participant_module_path=FLAGS.participant_module_path,
    )
    submission.run_benchmark()


if __name__ == '__main__':
    app.run(main)
