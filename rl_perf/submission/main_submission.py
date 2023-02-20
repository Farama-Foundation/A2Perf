import importlib.util
import os

from absl import app
import logging
import gin
from rl_perf.submission.submission_util import Submission
from rl_perf.submission.submission_util import BenchmarkDomain
from rl_perf.submission.submission_util import BenchmarkMode


def main(_):
    gin.parse_config_file('configs/base_config.gin')

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    submission = Submission()
    submission.run_benchmark()


if __name__ == '__main__':
    app.run(main)
