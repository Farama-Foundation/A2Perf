import os
from absl import app
import importlib.util


def run_benchmark(participant_module):
    pass


def main(_):
    location = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rlperf_benchmark_submission'))
    spec = importlib.util.spec_from_file_location("train", location)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    train.main()


if __name__ == '__main__':
    app.run(main)
