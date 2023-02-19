import importlib.util
import os

from absl import app
import logging
from submission import Submission


def main(_):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    participant_module_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'rlperf_benchmark_submission', 'train.py'))
    participant_module_spec = importlib.util.spec_from_file_location("train", participant_module_path)
    participant_module = importlib.util.module_from_spec(participant_module_spec)

    submission = Submission(participant_module=participant_module, participant_module_path=participant_module_path,
                            participant_module_spec=participant_module_spec,
                            benchmark='web_navigation', mode='train')
    submission.run_benchmark()


if __name__ == '__main__':
    app.run(main)
