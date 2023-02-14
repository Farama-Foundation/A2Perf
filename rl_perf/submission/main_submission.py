import os
from absl import app
import importlib.util
from rl_perf import profiler


def run_benchmark(participant_module):


    with profiler.TrainingMemoryProfiler() as mem_profiler:

        pass


def main(_):
    location = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rlperf_benchmark_submission'))
    spec = importlib.util.spec_from_file_location("train", location)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    train.main()


if __name__ == '__main__':
    app.run(main)
    # import the 'train' module from ../rlperf_benchmark_submission
