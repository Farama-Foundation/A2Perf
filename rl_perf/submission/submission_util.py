import enum
import importlib
import json
import logging
import multiprocessing
import os
import timeit
import typing
import math
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

from contextlib import contextmanager


@contextmanager
def working_directory(path):
    """Context manager for temporarily changing the working directory."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


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
    TRAINING_TIME = 'TrainingTime'
    MEMORY_USAGE = 'MemoryUsage'


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


def _start_inference_profilers(participant_event, profilers, pipes, profiler_started_events):
    processes = []
    profiler_objects = []
    for profiler_class, pipe, profiler_event in zip(profilers, pipes, profiler_started_events):
        logging.info(f'Starting profiler: {profiler_class}')
        profiler = profiler_class(pipe_for_participant_process=pipe, profiler_event=profiler_event,
                                  participant_process_event=participant_event)
        profiler_objects.append(profiler)
        profiler_process = multiprocessing.Process(target=profiler.start, )
        profiler_process.start()
        processes.append(profiler_process)
    return processes, profiler_objects


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
        self.participant_module_dir = os.path.dirname(self.participant_module_path)
        self.domain = domain
        self.mode = mode
        self.reliability_metrics = reliability_metrics
        self.metrics_results = {}

    def _load_participant_module(self, function_name):
        """Load the participant module and return the module object."""
        participant_module_dir = os.path.dirname(self.participant_module_path)
        original_dir = os.getcwd()
        os.chdir(participant_module_dir)
        participant_module_spec = importlib.util.spec_from_file_location(function_name, self.participant_module_path)
        participant_module = importlib.util.module_from_spec(participant_module_spec)
        os.chdir(original_dir)
        return participant_module, participant_module_spec

    @gin.configurable("Submission.create_domain")
    def create_domain(self, **kwargs):
        if self.domain == BenchmarkDomain.WEB_NAVIGATION:
            env = gym.make(self.domain.value, **kwargs)
        else:
            raise NotImplementedError(f'Domain {self.domain} not implemented')
        return env

    def _get_observation_data(self):
        if self.domain == BenchmarkDomain.WEB_NAVIGATION:
            env = self.create_domain()
            data = []
            for _ in range(self.num_inference_steps):
                observation = env.observation_space.sample()
                data.append(observation)
        else:
            raise NotImplementedError
        return data

    def _train(self, participant_event: multiprocessing.Event, profiler_events: typing.List[multiprocessing.Event]):
        participant_event.set()

        # Wait for all profilers to start up before continuing
        for profiler_event in profiler_events:
            profiler_event.wait()

        participant_module_spec = importlib.util.spec_from_file_location("train", self.participant_module_path)
        participant_module = importlib.util.module_from_spec(participant_module_spec)
        participant_module_spec.loader.exec_module(participant_module)
        participant_module.train()
        participant_event.clear()

    def _infer_async(self, participant_event: multiprocessing.Event, num_observations: int,
                     observation_pipe: multiprocessing.Pipe, ):
        # Profilers start after participant event is set
        participant_event.set()

        participant_module, participant_module_spec = self._load_participant_module("infer")
        with working_directory(self.participant_module_dir):
            participant_module_spec.loader.exec_module(participant_module)

        model = participant_module.load_model()
        counter = 0
        while counter < num_observations:
            observation = observation_pipe.recv()
            participant_module.infer_once(model, observation)
            counter += 1

        participant_event.clear()

    def _run_reliability_metrics(self):
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

    def _run_inference_benchmark_async(self):
        """Run inference benchmark with asynchronous observation sending to participant process.

        This method creates a participant process and a number of profiler processes. The participant process
        runs the participant module's `infer_once` method on each observation.

        Returns:
            profiler_objects: List of profiler objects. Use these objects to load the profiler results.
        """
        # Create pipes to communicate with the profilers
        pipes = [multiprocessing.Pipe() for _ in self.profilers]
        pipes_local, pipes_profiler = zip(*pipes)

        # Create a pipe to send observations to the participant process
        observation_pipe_local, observation_pipe_profiler = multiprocessing.Pipe()

        # Create event to signal to profilers that participant has started
        participant_started_event = multiprocessing.Event()
        profiler_started_events = [multiprocessing.Event() for _ in self.profilers]

        profiler_processes, profiler_objects = _start_inference_profilers(participant_event=participant_started_event,
                                                                          profilers=self.profilers,
                                                                          profiler_started_events=profiler_started_events,
                                                                          pipes=pipes_profiler)
        participant_process = multiprocessing.Process(target=self._infer_async,
                                                      args=(participant_started_event, self.num_inference_steps,
                                                            observation_pipe_profiler))
        participant_process.start()
        logging.info(f'Participant module process ID: {participant_process.pid}')
        for pipe in pipes_local:
            pipe.send(participant_process.pid)

        # Send observations to the participant process
        for observation in self._get_observation_data():
            observation_pipe_local.send(observation)

        participant_process.join()
        logging.info(f'Participant module process {participant_process.pid} finished')

        for profiler_process, profiler_object in zip(profiler_processes, profiler_objects):
            profiler_process.join()
            logging.info(f'Profiler process {profiler_process.pid} finished')
        return profiler_objects

    def _run_inference_benchmark(self):
        ##################################################
        # Setting up the participants code
        ##################################################

        # Load the participant module
        participant_module, participant_module_spec = self._load_participant_module("infer")
        with working_directory(self.participant_module_dir):
            participant_module_spec.loader.exec_module(participant_module)

        # Get the participant model
        participant_model = participant_module.load_model()

        ##################################################
        # Timing metrics for single inference steps
        ##################################################
        inference_data = self._get_observation_data()

        def inference_step():
            return participant_module.infer_once(model=participant_model, observation=inference_data[i])

        if self.time_inference_steps:
            inference_times = []
            for i in range(self.num_inference_steps):
                inference_times.append(timeit.timeit(inference_step, number=1))

            self.metrics_results['inference_time'] = dict(values=inference_times,
                                                          mean=np.mean(inference_times),
                                                          std=np.std(inference_times))

        ##################################################
        # Asynchronous inference metrics
        ##################################################
        done_profiler_objects = []
        if self.profilers:
            done_profiler_objects = self._run_inference_benchmark_async()

        if self.reliability_metrics:
            self._run_reliability_metrics()

        ##################################################
        # Save raw metrics to disk, and plot results
        ##################################################

        if not os.path.exists(os.path.join(self.base_log_dir, 'metrics')):
            os.makedirs(os.path.join(self.base_log_dir, 'metrics'))
        with open(os.path.join(self.base_log_dir, 'metrics', 'metric_results.json'), 'w') as f:
            json.dump(self.metrics_results, f)

        # Plot metrics and save to file
        if self.plot_metrics:
            for profiler_object in done_profiler_objects:
                title, fig = profiler_object.plot_results()
                plt.savefig(os.path.join(self.base_log_dir, 'metrics', f'{title}.png'))
            self._plot_metrics()

    def _plot_metrics(self):
        if not self.metrics_results:
            logging.warning('No metrics to plot')

        for metric_name, metric_values in self.metrics_results.items():
            plt.figure()
            plt.title(metric_name)
            plt.plot(metric_values['values'])
            plt.savefig(os.path.join(self.metric_values_dir, f'{metric_name}.png'))

    def run_benchmark(self):
        if self.mode == BenchmarkMode.TRAIN:
            self._run_training_benchmark()
        elif self.mode == BenchmarkMode.INFERENCE:
            self._run_inference_benchmark()
        else:
            raise ValueError('Benchmark mode must be either train or infer')
