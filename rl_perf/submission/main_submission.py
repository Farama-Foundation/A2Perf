import logging
import multiprocessing

import gin
from absl import app

from rl_perf.submission.submission_util import Submission


def main(_):
    # TODO: make sure spawn doesn't break anything.
    multiprocessing.set_start_method('spawn', force=True)

    gin.parse_config_file('configs/base_config.gin')

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    submission = Submission()
    submission.run_benchmark()


if __name__ == '__main__':
    app.run(main)
