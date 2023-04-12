import enum
import importlib
import json
import logging
import multiprocessing
import os
import timeit
import typing

import gin
import gym
import matplotlib.pyplot as plt
import numpy as np
from rl_perf.domains import web_nav

from rl_perf.metrics.profiler.base_profiler import BaseProfiler
from rl_perf.metrics.reliability.rl_reliability_metrics.evaluation.eval_metrics import Evaluator
from rl_perf.metrics.reliability.rl_reliability_metrics.metrics import (IqrAcrossRuns, IqrWithinRuns,
                                                                        LowerCVaROnAcross,
                                                                        LowerCVaROnDiffs, )


@gin.constants_from_enum
class BenchmarkMode(enum.Enum):
    TRAIN = 'train'
    INFERENCE = 'inference'


@gin.constants_from_enum
class BenchmarkDomain(enum.Enum):
    WEB_NAVIGATION = 'WebNavigation-v0'
    CIRCUIT_TRAINING = 'CircuitTraining-v0'
    QUADRUPED_LOCOMOTION = 'QuadrupedLocomotion-v0'


@gin.constants_from_enum
class SystemMetrics(enum.Enum):
    INFERENCE_TIME = 'InferenceTime'


@gin.constants_from_enum
class ReliabilityMetrics(enum.Enum):
    IqrWithinRuns = 'IqrWithinRuns'
    IqrAcrossRuns = 'IqrAcrossRuns'
    LowerCVaROnDiffs = 'LowerCVaROnDiffs'
    LowerCVaROnDrawdown = 'LowerCVaROnDrawdown'
    LowerCVarOnAcross = 'LowerCVarOnAcross'
    MedianPerfDuringTraining = 'MedianPerfDuringTraining'


def _start_profilers(profilers: typing.List[typing.Type[BaseProfiler]],
                     profiler_events: typing.List[multiprocessing.Event], participant_event: multiprocessing.Event,
                     participant_process: multiprocessing.Process, log_dir: str) -> \
        typing.List[multiprocessing.Process]:
    processes = []
    for profiler_class, profiler_event in zip(profilers, profiler_events):
        logging.info(f'Starting profiler: {profiler_class}')
        profiler = profiler_class(participant_process_event=participant_event,
                                  profiler_event=profiler_event,
                                  participant_process=participant_process,
                                  base_log_dir=log_dir)
        profiler_process = multiprocessing.Process(target=profiler.start)
        profiler_process.start()
        processes.append(profiler_process)
    return processes


@gin.configurable
class Submission:
    def __init__(self,
                 participant_module_path: str = None,
                 profilers: typing.List[typing.Type[BaseProfiler]] = None,
                 mode: BenchmarkMode = BenchmarkMode.TRAIN,
                 domain: BenchmarkDomain = BenchmarkDomain.WEB_NAVIGATION,
                 base_log_dir: str = None,
                 metric_values_dir: str = None,
                 num_inference_steps: int = 1000,
                 time_participant_code: bool = True,
                 plot_metrics: bool = True,
                 reliability_metrics: typing.List[ReliabilityMetrics] = None):

        self.base_log_dir = base_log_dir
        if self.base_log_dir is not None:
            os.makedirs(self.base_log_dir, exist_ok=True)

        self.metric_values_dir = metric_values_dir
        self.plot_metrics = plot_metrics
        if self.metric_values_dir is not None:
            os.makedirs(self.metric_values_dir, exist_ok=True)
        self.num_inference_steps = num_inference_steps
        self.time_inference_steps = time_participant_code
        self.profilers = profilers if profilers is not None else []
        self.participant_module_path = participant_module_path
        self.domain = domain
        self.mode = mode
        self.reliability_metrics = reliability_metrics
        self.metrics_results = {}

    def _train(self, participant_event, profiler_events):
        participant_event.set()

        # Wait for all profilers to start up before continuing
        for profiler_event in profiler_events:
            profiler_event.wait()

        participant_module_spec = importlib.util.spec_from_file_location("train", self.participant_module_path)
        participant_module = importlib.util.module_from_spec(participant_module_spec)
        participant_module_spec.loader.exec_module(participant_module)
        participant_module.train()
        participant_event.clear()

    def _infer(self, participant_event, profiler_events):

        # Wait for all profilers to start up before continuing with inference
        for profiler_event in profiler_events:
            profiler_event.wait()

        participant_module_spec = importlib.util.spec_from_file_location("infer", self.participant_module_path)
        participant_module = importlib.util.module_from_spec(participant_module_spec)
        participant_module_spec.loader.exec_module(participant_module)

        # Get the participant model
        participant_model = participant_module.load_model()

        # Since all the startup is done, we can communicate to the profilers that they can start profiling
        participant_event.set()

        participant_event.clear()

        # TODO: Add a check to make sure that the participant's actions are valid

    def run_reliability_metrics(self):
        # TODO make sure to write gin config for metric parameters
        metrics = []
        for metric in self.reliability_metrics:
            if metric == ReliabilityMetrics.IqrWithinRuns:
                metrics.append(IqrWithinRuns())
            elif metric == ReliabilityMetrics.IqrAcrossRuns:
                metrics.append(IqrAcrossRuns())
            elif metric == ReliabilityMetrics.LowerCVaROnDiffs:
                metrics.append(LowerCVaROnDiffs())
            elif metric == ReliabilityMetrics.LowerCVaROnDrawdown:
                metrics.append(LowerCVaROnDiffs())
            elif metric == ReliabilityMetrics.LowerCVarOnAcross:
                metrics.append(LowerCVaROnAcross())
            elif metric == ReliabilityMetrics.MedianPerfDuringTraining:
                raise NotImplementedError('MedianPerfDuringTraining not implemented yet')
            else:
                raise ValueError(f'Invalid metric: {metric}')

        run_paths = [os.path.join(self.base_log_dir, run_path, 'train', 'train_summary.csv') for run_path in
                     os.listdir(self.base_log_dir)]

        # TODO:
        #   define n_timeframes in gin (for dividing each run into timeframes)
        #   define n_random_samples for permutations /bootstraps
        #   define n_worker for permutations /bootstraps
        #   define pvals_dir in gin
        #   define confidence_intervals_dir in gin
        #   define plots_dir in gin
        #   define tasks in gin (for locomotion this would be the different gaits)
        #   define the algorithms in gin (for locomotion this would be MPC, PPO, etc)
        #   define n_runs_per_experiment in gin

        evaluator = Evaluator(metrics=metrics, )
        self.metrics_results['reliability_metrics'] = evaluator.evaluate(run_paths=run_paths,
                                                                         outfile_prefix=self.metric_values_dir)

    def _run_training_benchmark(self):
        # Need a participant event to signal to profilers
        participant_started_event = multiprocessing.Event()
        profilers_started_events = [multiprocessing.Event() for _ in self.profilers]

        participant_process = multiprocessing.Process(target=self._train,
                                                      args=(participant_started_event, profilers_started_events))
        participant_process.start()
        profilers = _start_profilers(profilers=self.profilers, participant_event=participant_started_event,
                                     profiler_events=profilers_started_events,
                                     participant_process=participant_process, log_dir=self.base_log_dir)
        logging.info(f'Participant module process ID: {participant_process.pid}')
        participant_process.join()
        logging.info(f'Participant module process {participant_process.pid} finished')

        for profiler in profilers:
            profiler.join()
            logging.info(f'Profiler process {profiler.pid} finished')

    def _run_inference_benchmark(self):
        ##################################################
        # Setting up the participants code
        ##################################################

        # Load participant module. Change the current working directory to the directory containing the participant
        # module file, so that any relative imports work
        participant_module_dir = os.path.dirname(self.participant_module_path)
        original_dir = os.getcwd()
        os.chdir(participant_module_dir)
        participant_module_spec = importlib.util.spec_from_file_location("infer", self.participant_module_path)
        participant_module = importlib.util.module_from_spec(participant_module_spec)
        participant_module_spec.loader.exec_module(participant_module)
        os.chdir(original_dir)  # Change back to the original working directory

        # Generate fake data for inference
        # TODO: replace this with a function to create environments since we need different args for different domains
        env = gym.make(self.domain.value, difficulty=1, seed=0)
        data = []
        for _ in range(self.num_inference_steps):
            observation = env.observation_space.sample()
            preprocessed_obs = participant_module.preprocess_observation(observation)
            data.append(preprocessed_obs)

        # Get the participant model
        participant_model = participant_module.load_model()

        ##################################################
        # Timing metrics for single inference steps
        ##################################################
        def inference_step():
            return participant_module.infer_once(model=participant_model, observation=data[i])

        if self.time_inference_steps:
            inference_times = []
            for i in range(self.num_inference_steps):
                inference_times.append(timeit.timeit(inference_step, number=1))

            self.metrics_results['inference_time'] = dict(values=inference_times,
                                                          mean=np.mean(inference_times),
                                                          std=np.std(inference_times))

        # TODO: Run other metrics for inference

        if self.reliability_metrics:
            self.run_reliability_metrics()

        ##################################################
        # Save raw metrics to disk, and plot results
        ##################################################

        if not os.path.exists(os.path.join(self.base_log_dir, 'metrics')):
            os.makedirs(os.path.join(self.base_log_dir, 'metrics'))
        with open(os.path.join(self.base_log_dir, 'metrics', 'metric_results.json'), 'w') as f:
            json.dump(self.metrics_results, f)

        # Plot metrics and save to file
        if self.plot_metrics:
            self._plot_metrics()

    def _plot_metrics(self):
        if not self.metrics_results:
            logging.warning('No metrics to plot')

        if 'inference_time' in self.metrics_results:
            fig, ax = plt.subplots()
            ax.boxplot(self.metrics_results['inference_time']['values'], labels=['DQNLSTM (TFA)'])
            ax.set_title('Inference Time')
            ax.set_ylabel('Time (s)')
            fig.savefig(os.path.join(self.base_log_dir, 'metrics', 'inference_times.png'))

            fig, ax = plt.subplots()
            ax.plot(self.metrics_results['inference_time']['values'], label='DQNLSTM (TFA)')
            ax.set_title('Inference Time')
            ax.set_ylabel('Time (s)')
            ax.set_xlabel('Step')
            ax.legend(loc='upper right')
            fig.savefig(os.path.join(self.base_log_dir, 'metrics', 'inference_times_line.png'))

    def run_benchmark(self):
        if self.mode == BenchmarkMode.TRAIN:
            self._run_training_benchmark()
        elif self.mode == BenchmarkMode.INFERENCE:
            self._run_inference_benchmark()
        else:
            raise ValueError('Benchmark mode must be either train or infer')
