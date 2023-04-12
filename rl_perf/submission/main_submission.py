import logging
import multiprocessing
import os

import gin
from absl import app
from absl import flags
from rl_perf.submission.submission_util import Submission

flags.DEFINE_string('gin_file', None, 'Path to the gin-config file.')
FLAGS = flags.FLAGS


def main(_):
    # Set the working directory to the submission directory.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # TODO: make sure spawn doesn't break anything.
    multiprocessing.set_start_method('spawn', force=True)

    gin.parse_config_file(FLAGS.gin_file)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    submission = Submission()
    submission.run_benchmark()


if __name__ == '__main__':
    app.run(main)
