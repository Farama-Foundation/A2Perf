import csv
import enum
import importlib
import json
import logging
import multiprocessing
import os
import sys
import timeit
import typing
from contextlib import contextmanager

import gin
import gym
import matplotlib.pyplot as plt
import numpy as np

from rl_perf.domains.web_nav.CoDE import vocabulary_node
from rl_perf.metrics.reliability.rl_reliability_metrics.evaluation.eval_metrics import Evaluator
from rl_perf.metrics.reliability.rl_reliability_metrics.metrics import (IqrAcrossRuns, IqrWithinRuns,
                                                                        LowerCVaROnAcross,
                                                                        LowerCVaROnDiffs,
                                                                        IqrAcrossRollouts, MedianPerfDuringTraining
, StddevAcrossRollouts, MadAcrossRollouts, UpperCVaRAcrossRollouts, LowerCVaRAcrossRollouts, )
from rl_perf.metrics.system import codecarbon
from rl_perf.metrics.system.profiler.base_profiler import BaseProfiler


@contextmanager
def working_directory(path):
    """Context manager for temporarily changing the working directory."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        sys.path.remove(path)


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
    MadAcrossRollouts = 'MadAcrossRollouts'
    IqrAcrossRollouts = 'IqrAcrossRollouts'
    StddevAcrossRollouts = 'StddevAcrossRollouts'
    UpperCVaRAcrossRollouts = 'UpperCVaRAcrossRollouts'
    LowerCVaRAcrossRollouts = 'LowerCVaRAcrossRollouts'


def _start_profilers(profilers: typing.List[typing.Type[BaseProfiler]],
                     profiler_events: typing.List[multiprocessing.Event], participant_event: multiprocessing.Event,
                     participant_process: multiprocessing.Process, log_dir: str, mp_context) -> \
        typing.List[multiprocessing.Process]:
    processes = []
    for profiler_class, profiler_event in zip(profilers, profiler_events):
        logging.info(f'Starting profiler: {profiler_class}')
        profiler = profiler_class(participant_process_event=participant_event,
                                  profiler_event=profiler_event,
                                  participant_process=participant_process,
                                  base_log_dir=log_dir)
        profiler_process = mp_context.Process(target=profiler.start)
        profiler_process.start()
        processes.append(profiler_process)
    return processes


def _start_inference_profilers(participant_event, profilers, pipes, profiler_started_events, base_log_dir, mp_context):
    processes = []
    profiler_objects = []
    for profiler_class, pipe, profiler_event in zip(profilers, pipes, profiler_started_events):
        logging.info(f'Starting profiler: {profiler_class}')
        profiler = profiler_class(pipe_for_participant_process=pipe, profiler_event=profiler_event,
                                  participant_process_event=participant_event, base_log_dir=base_log_dir)
        profiler_objects.append(profiler)
        profiler_process = mp_context.Process(target=profiler.start, )
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
                 root_dir: str = None,
                 metric_values_dir: str = None,
                 train_logs_dirs: typing.List[str] = None,
                 num_inference_steps: int = 1000,
                 num_inference_episodes: int = 1,
                 time_participant_code: bool = True,
                 measure_emissions: bool = False,
                 baseline_measure_sec: float = 0,
                 plot_metrics: bool = True,
                 run_offline_metrics_only: bool = False,
                 reliability_metrics: typing.List[ReliabilityMetrics] = None,
                 tracking_mode: str = None):
        self.run_offline_metrics_only = run_offline_metrics_only
        self.baseline_measure_sec = baseline_measure_sec
        self.tracking_mode = tracking_mode
        self.mp_context = multiprocessing.get_context('spawn')
        self.root_dir = root_dir
        if self.root_dir is not None:
            os.makedirs(self.root_dir, exist_ok=True)
        self.gin_config_str = None
        self.metric_values_dir = metric_values_dir
        self.train_logs_dirs = train_logs_dirs
        if self.metric_values_dir is None:
            self.metric_values_dir = os.path.join(self.root_dir, 'metrics')
        if self.train_logs_dirs is None:
            self.train_logs_dirs = os.path.join(self.root_dir, 'train')
        os.makedirs(self.metric_values_dir, exist_ok=True)
        for dir in self.train_logs_dirs:
            os.makedirs(dir, exist_ok=True)
        self.measure_emissions = measure_emissions
        self.plot_metrics = plot_metrics
        self.num_inference_steps = num_inference_steps
        self.num_inference_episodes = num_inference_episodes
        self.time_inference_steps = time_participant_code
        self.profilers = profilers if profilers is not None else []
        for profiler in self.profilers:
            profiler.base_log_dir = self.root_dir
        self.participant_module_path = os.path.abspath(participant_module_path)
        self.domain = domain
        self.mode = mode
        self.reliability_metrics = reliability_metrics

        self.metrics_results = {}
        if os.path.exists(os.path.join(self.metric_values_dir, 'metric_results.json')):
            logging.info(
                f'Loading pre-existing metric results from {os.path.join(self.metric_values_dir, "metric_results.json")}')
            with open(os.path.join(self.metric_values_dir, 'metric_results.json'), 'r') as f:
                self.metrics_results = json.load(f)

    def _load_participant_spec(self, filename):
        """Loads the participant spec from the participant module path."""
        participant_file_path = os.path.join(self.participant_module_path, filename)
        spec = importlib.util.spec_from_file_location(f"{filename}", participant_file_path)
        return spec

    def _load_participant_module(self, filename):
        """Load the participant module and return the module object."""
        spec = self._load_participant_spec(filename)
        participant_module = importlib.util.module_from_spec(spec)
        return participant_module, spec

    @gin.configurable("Submission.create_domain")
    def create_domain(self, **kwargs):
        if self.domain == BenchmarkDomain.WEB_NAVIGATION:
            if kwargs.get('reload_vocab', False):
                global_vocab_dict = np.load(os.path.join(self.root_dir, 'train', 'global_vocab.npy'),
                                            allow_pickle=True).item()
                global_vocab = vocabulary_node.LockedVocabulary()
                global_vocab.restore(dict(global_vocab=global_vocab_dict))
                kwargs['global_vocabulary'] = global_vocab
                kwargs.pop('reload_vocab')
        elif self.domain == BenchmarkDomain.CIRCUIT_TRAINING:
            pass
        elif self.domain == BenchmarkDomain.QUADRUPED_LOCOMOTION:
            env = None
        else:
            raise NotImplementedError(f'Domain {self.domain} not implemented')
        env = gym.make(self.domain.value, **kwargs)
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
        gin.parse_config(self.gin_config_str)
        participant_event.set()

        # Wait for all profilers to start up before continuing
        for profiler_event in profiler_events:
            profiler_event.wait()

        with working_directory(self.participant_module_path):
            participant_module, participant_module_spec = self._load_participant_module('train.py')
            print(self.participant_module_path)
            print(participant_module_spec)
            participant_module_spec.loader.exec_module(participant_module)

            if self.measure_emissions:
                @codecarbon.track_emissions(output_dir=self.metric_values_dir, output_file='train_emissions.csv', )
                def train():
                    return participant_module.train()

                train()
            else:
                participant_module.train()

        participant_event.clear()
        for profiler_event in profiler_events:
            profiler_event.clear()

    def _infer_async(self, participant_event: multiprocessing.Event, num_observations: int,
                     observation_pipe: multiprocessing.Pipe, ):
        # Profilers start after participant event is set
        participant_event.set()

        with working_directory(self.participant_module_path):
            participant_module, participant_module_spec = self._load_participant_module('inference.py')
            participant_module_spec.loader.exec_module(participant_module)

            model = participant_module.load_model()
            counter = 0
            while counter < num_observations:
                observation = observation_pipe.recv()
                participant_module.infer_once(model, observation)
                counter += 1

            participant_event.clear()

    def _run_inference_reliability_metrics(self, values=None):
        metrics = []
        for metric in self.reliability_metrics:
            if metric == ReliabilityMetrics.MadAcrossRollouts:
                metrics.append(MadAcrossRollouts())
            elif metric == ReliabilityMetrics.IqrAcrossRollouts:
                metrics.append(IqrAcrossRollouts())
            elif metric == ReliabilityMetrics.UpperCVaRAcrossRollouts:
                metrics.append(UpperCVaRAcrossRollouts())
            elif metric == ReliabilityMetrics.LowerCVaRAcrossRollouts:
                metrics.append(LowerCVaRAcrossRollouts())
            elif metric == ReliabilityMetrics.StddevAcrossRollouts:
                metrics.append(StddevAcrossRollouts())
            elif metric == ReliabilityMetrics.UpperCVaRAcrossRollouts:
                metrics.append(UpperCVaRAcrossRollouts())
            elif metric == ReliabilityMetrics.LowerCVaRAcrossRollouts:
                metrics.append(LowerCVaRAcrossRollouts())
            else:
                raise ValueError(f'Invalid metric: {metric}')
        with open(os.path.join(self.metric_values_dir, 'rollouts.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_num', 'reward'])
            for i, value in enumerate(values):
                writer.writerow([i, value])

        evaluator = Evaluator(metrics=metrics, )
        reliability_metrics = evaluator.evaluate(
            run_paths=[os.path.join(self.metric_values_dir, 'rollouts.csv')], )
        self.metrics_results.update(reliability_metrics)

    def _run_train_reliability_metrics(self):
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
                metrics.append(MedianPerfDuringTraining())
            else:
                raise ValueError(f'Invalid metric: {metric}')

        logging.info(f'Running reliability metrics: {metrics}')
        logging.info(f'Logging to {self.metric_values_dir}')

        # TODO: match with regex
        run_paths = self.train_logs_dirs
        logging.info(f'Found {len(run_paths)} runs in {self.train_logs_dirs}')
        logging.info(f'Run paths: {run_paths}')
        evaluator = Evaluator(metrics=metrics, )
        reliability_metrics = evaluator.evaluate(run_paths=run_paths, )
        self.metrics_results.update(reliability_metrics)

    def _run_training_benchmark(self):
        if not self.run_offline_metrics_only:
            # Need a participant event to signal to profilers
            participant_started_event = multiprocessing.Event()
            profilers_started_events = [multiprocessing.Event() for _ in self.profilers]

            participant_process = self.mp_context.Process(target=self._train,
                                                          args=(participant_started_event, profilers_started_events))
            participant_process.start()
            profilers = _start_profilers(profilers=self.profilers, participant_event=participant_started_event,
                                         profiler_events=profilers_started_events,
                                         participant_process=participant_process,
                                         log_dir=self.root_dir,
                                         mp_context=self.mp_context)
            logging.info(f'Participant module process ID: {participant_process.pid}')
            participant_process.join()
            logging.info(f'Participant module process {participant_process.pid} finished')
            if participant_process.is_alive():
                logging.error('Participant process is still running')
            elif participant_process.exitcode != 0:
                logging.error(f'Participant process exited with code {participant_process.exitcode}')
            else:
                logging.info(f'Participant process {participant_process.pid} finished')

            for profiler in profilers:
                profiler.join()
                if profiler.is_alive():
                    logging.error(f'Profiler process {profiler.pid} is still running')
                elif profiler.exitcode != 0:
                    logging.error(f'Profiler process {profiler.pid} exited with code {profiler.exitcode}')
                else:
                    logging.info(f'Profiler process {profiler.pid} finished')

        if self.reliability_metrics:
            self._run_train_reliability_metrics()

            ##################################################
            # Save raw metrics to disk, and plot results
            ##################################################

            with open(os.path.join(self.metric_values_dir, 'metric_results.json'), 'w') as f:
                json.dump(self.metrics_results, f)

            # Plot metrics and save to file
            if self.plot_metrics:
                for profiler_object in self.profilers:
                    title, fig = profiler_object.plot_results()
                    plt.savefig(os.path.join(self.metric_values_dir, f'{title}.png'))
                self._plot_metrics()

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
                                                                          pipes=pipes_profiler,
                                                                          base_log_dir=self.metric_values_dir,
                                                                          mp_context=self.mp_context)
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

        # Clear events
        for event in profiler_started_events:
            event.clear()
        participant_started_event.clear()

        return profiler_objects

    def _run_inference_benchmark(self):
        ##################################################
        # Setting up the participants code
        ##################################################

        # Load the participant module
        participant_module, participant_module_spec = self._load_participant_module('inference.py')

        with working_directory(self.participant_module_path):
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
        # Hardware energy consumption using codecarbon
        ##################################################
        if self.measure_emissions:
            @codecarbon.track_emissions(project_name='rlperf_inference',
                                        output_dir=self.metric_values_dir,
                                        output_file='inference_emissions.csv',
                                        save_to_file=True,
                                        save_to_api=False,
                                        save_to_logger=False,
                                        api_call_interval=1,
                                        country_iso_code=self.country_iso_code,
                                        region=self.region,
                                        offline=self.code_carbon_offline_mode,
                                        tracking_mode=self.tracking_mode,
                                        measure_power_secs=self.measure_emissions_interval,
                                        baseline_measure_sec=self.baseline_measure_sec,

                                        )
            def measure_emissions():
                for i in range(self.num_inference_steps):
                    # choose a random observation
                    observation = inference_data[i]
                    participant_module.infer_once(model=participant_model, observation=observation)

            measure_emissions()

        ##################################################
        # Asynchronous inference metrics
        ##################################################
        done_profiler_objects = []
        if self.profilers:
            done_profiler_objects = self._run_inference_benchmark_async()

        ##################################################
        # Perform rollouts with the participant module
        ##################################################
        all_rewards = []
        env = self.create_domain()
        for _ in range(self.num_inference_episodes):
            observation = env.reset()
            done = False
            rewards = 0
            while not done:
                action = participant_module.infer_once(model=participant_model, observation=observation)
                observation, reward, done, _ = env.step(action)
                rewards += reward
            all_rewards.append(rewards)
            print(f'Episode reward: {rewards}')
        self.metrics_results['episode_rewards'] = dict(values=all_rewards,
                                                       mean=np.mean(all_rewards),
                                                       std=np.std(all_rewards))

        ##################################################
        # Run reliability metrics
        ##################################################
        if self.reliability_metrics:
            self._run_inference_reliability_metrics(values=all_rewards)

        ##################################################
        # Save raw metrics to disk, and plot results
        ##################################################

        with open(os.path.join(self.metric_values_dir, 'metric_results.json'), 'w') as f:
            json.dump(self.metrics_results, f)

        # Plot metrics and save to file
        if self.plot_metrics:
            for profiler_object in done_profiler_objects:
                title, fig = profiler_object.plot_results()
                plt.savefig(os.path.join(self.metric_values_dir, f'{title}.png'))
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
        self.gin_config_str = gin.config_str()  # save gin configs for multiprocessing
        if self.mode == BenchmarkMode.TRAIN:
            self._run_training_benchmark()
        elif self.mode == BenchmarkMode.INFERENCE:
            self._run_inference_benchmark()
        else:
            raise ValueError('Benchmark mode must be either train or inference')
