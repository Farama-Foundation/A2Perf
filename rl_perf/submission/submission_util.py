import collections
import csv
import enum
import gin
import glob
import gymnasium as gym
import importlib
import json
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import sys
import timeit
import typing
from contextlib import contextmanager
from decimal import Decimal

from rl_perf.metrics.reliability.rl_reliability_metrics.evaluation.eval_metrics import Evaluator
from rl_perf.metrics.reliability.rl_reliability_metrics.metrics import (IqrAcrossRuns, IqrWithinRuns,
                                                                        LowerCVaROnAcross,
                                                                        LowerCVaROnDiffs,
                                                                        LowerCVaROnDrawdown,
                                                                        UpperCVaROnDrawdown,
                                                                        IqrAcrossRollouts, MedianPerfDuringTraining
, StddevAcrossRollouts, MadAcrossRollouts, UpperCVaRAcrossRollouts, LowerCVaRAcrossRollouts, )
from rl_perf.metrics.system import codecarbon
from rl_perf.metrics.system.profiler.base_profiler import BaseProfiler

SCI_NOTATION_THRESHOLD = 1e-2


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
    # Train
    IqrWithinRuns = 'IqrWithinRuns'
    IqrAcrossRuns = 'IqrAcrossRuns'
    LowerCVaROnDiffs = 'LowerCVaROnDiffs'
    LowerCVarOnAcross = 'LowerCVarOnAcross'
    UpperCVaROnDrawdown = 'UpperCVaROnDrawdown'

    # Inference
    IqrAcrossRollouts = 'IqrAcrossRollouts'
    LowerCVaRAcrossRollouts = 'LowerCVaRAcrossRollouts'


@gin.configurable
def collect_system_metrics(emissions_file_paths: typing.List[str],
                           benchmark_mode: BenchmarkMode,
                           metric_results: typing.Dict[str, typing.Any] = None,
                           subset_size: int = None) -> typing.Dict[str, typing.Any]:
    metric_values_dict = {
        "cpu_energy": {"values": [], "mean": None, "std": None, "units": "kWh"},
        "gpu_energy": {"values": [], "mean": None, "std": None, "units": "kWh"},
        "ram_energy": {"values": [], "mean": None, "std": None, "units": "kWh"},
        "cpu_power": {"values": [], "mean": None, "std": None, "units": "W"},
        "gpu_power": {"values": [], "mean": None, "std": None, "units": "W"},
        "ram_power": {"values": [], "mean": None, "std": None, "units": "W"},
        "ram_process": {"values": [], "mean": None, "std": None, "units": "GB"},
        "all_energy": {"values": [], "mean": None, "std": None, "units": "kWh"},
        "all_power": {"values": [], "mean": None, "std": None, "units": "W"},
    }

    if benchmark_mode == BenchmarkMode.TRAIN:
        metric_values_dict["duration"] = {"values": [], "mean": None, "std": None, "units": "h"}
    elif benchmark_mode == BenchmarkMode.INFERENCE:
        metric_values_dict["inference_time"] = {"values": [], "mean": None, "std": None, "units": "s"}
        all_metric_values_dict["inference_time"] = {"values": [], "mean": None, "std": None, "units": "s"}
    else:
        raise ValueError(f'Unknown benchmark mode: {benchmark_mode}')

    for emissions_file_path in emissions_file_paths:
        logging.info(f'Collecting system metrics from {emissions_file_path}')
        df = pd.read_csv(emissions_file_path)
        df['all_energy'] = df['cpu_energy'] + df['gpu_energy'] + df['ram_energy']
        df['all_power'] = df['cpu_power'] + df['gpu_power'] + df['ram_power']
        df['total_energy'] = df['all_energy'].sum()
        df['total_power'] = df['all_power'].sum()
        df['total_cpu_energy'] = df['cpu_energy'].sum()
        df['total_gpu_energy'] = df['gpu_energy'].sum()
        df['total_ram_energy'] = df['ram_energy'].sum()
        df['total_cpu_power'] = df['cpu_power'].sum()
        df['total_gpu_power'] = df['gpu_power'].sum()
        df['total_ram_power'] = df['ram_power'].sum()

        for key in metric_values_dict.keys():
            if key == 'duration':
                if benchmark_mode == BenchmarkMode.TRAIN:
                    start_time = pd.to_datetime(df['timestamp'].iloc[0])
                    end_time = pd.to_datetime(df['timestamp'].iloc[-1])
                    duration = (end_time - start_time).total_seconds()
                    logging.info(f"Duration: {duration}")
                    duration = duration / 60 / 60  # convert to hours
                    metric_values_dict[key]["values"].append(duration)
            elif key == 'inference_time':
                if benchmark_mode == BenchmarkMode.INFERENCE:
                    inference_times = metric_results['inference_time']['values']
                    avg_inference_time = np.mean(inference_times)
                    if avg_inference_time < 1.0:
                        logging.info(f'Converting inference time from seconds to milliseconds')
                        # convert to milliseconds
                        inference_times = [i * 1000 for i in inference_times]
                        metric_values_dict[key]["units"] = "ms"
                        metric_values_dict[key]["values"] = inference_times
            else:
                metric_values_dict[key]["values"].append(np.squeeze(df[key].values))

    for key in metric_values_dict.keys():
        print('key: ', key)
        print('values: ', metric_values_dict[key]["values"])
        if key != 'duration':
            metric_values_dict[key]["values"] = np.concatenate(metric_values_dict[key]["values"])

        metric_values_dict[key]["mean"] = np.mean(metric_values_dict[key]["values"])
        metric_values_dict[key]["std"] = np.std(metric_values_dict[key]["values"])

        if subset_size is not None and key != 'duration':
            metric_values_dict[key]["values"] = np.random.choice(metric_values_dict[key]["values"], subset_size)
            metric_values_dict[key]["values"] = metric_values_dict[key]["values"].tolist()

    return metric_values_dict


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
                 root_dir: str,
                 participant_module_path: str = None,
                 profilers: typing.List[typing.Type[BaseProfiler]] = None,
                 mode: BenchmarkMode = BenchmarkMode.TRAIN,
                 domain: BenchmarkDomain = BenchmarkDomain.WEB_NAVIGATION,
                 domain_config_paths: str = None,
                 metric_values_dir: str = None,
                 train_logs_dirs: typing.List[str] = None,
                 num_inference_steps: int = 1000,
                 num_inference_episodes: int = 1,
                 time_participant_code: bool = True,
                 measure_emissions: bool = False,
                 baseline_measure_sec: float = 0,
                 plot_metrics: bool = False,
                 run_offline_metrics_only: bool = False,
                 reliability_metrics: typing.List[ReliabilityMetrics] = None,
                 tracking_mode: str = None):
        """Object that represents a submission to the benchmark.

        Args:
            participant_module_path: Path to the module that contains the participant's code.
            profilers: List of profilers to use.
            mode: Benchmark mode (train or inference).
            domain: Benchmark domain (web navigation, circuit training or quadruped locomotion).
            root_dir: Root directory for the submission.
            metric_values_dir: Directory where the metric values will be saved.
            train_logs_dirs: List of directories where the training logs are saved. Relative to the root directory.
            num_inference_steps: Number of steps to run inference for.
            num_inference_episodes: Number of episodes to run inference for.
            time_participant_code: Whether to time the participant's code.
            measure_emissions: Whether to measure emissions.
            baseline_measure_sec: Baseline time to measure emissions for.
            plot_metrics: Whether to plot the metrics.
            run_offline_metrics_only: Whether to run only the offline metrics.
            reliability_metrics: List of reliability metrics to compute.
            tracking_mode: Tracking mode for the participant's code.
        """
        self.root_dir = root_dir
        self.metric_values_dir = metric_values_dir
        self.train_logs_dirs = train_logs_dirs
        print(train_logs_dirs)
        os.makedirs(self.root_dir, exist_ok=True)
        if self.metric_values_dir is None:
            self.metric_values_dir = os.path.join(self.root_dir, 'metrics')
        os.makedirs(self.metric_values_dir, exist_ok=True)
        self.run_offline_metrics_only = run_offline_metrics_only
        self.baseline_measure_sec = baseline_measure_sec
        self.tracking_mode = tracking_mode
        self.mp_context = multiprocessing.get_context('spawn')
        self.gin_config_str = None
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
        metrics_path = os.path.join(self.metric_values_dir,
                                    'inference_metrics.json' if self.mode == BenchmarkMode.INFERENCE else 'train_metrics.json')
        if os.path.exists(metrics_path):
            logging.info(
                f'Loading pre-existing metric results from {metrics_path}')
            with open(metrics_path, 'r') as f:
                self.metrics_results = json.load(f)

        if domain_config_paths:
            self.domain_config_paths = glob.glob(domain_config_paths)
            if self.domain_config_paths:
                logging.info(f'Using domain config paths: {self.domain_config_paths}')
            else:
                logging.warning(f'No domain config paths found in {domain_config_paths}')
        else:
            self.domain_config_paths = []
            logging.warning(f'No domain config paths provided. Using default environment configuration')

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
            from rl_perf.domains.web_nav.CoDE import vocabulary_node

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
            from rl_perf.domains import quadruped_locomotion
        else:
            raise NotImplementedError(f'Domain {self.domain} not implemented')
        env = gym.make(self.domain.value, **kwargs)
        return env

    def _get_observation_data(self):
        env = self.create_domain()
        data = []
        for _ in range(self.num_inference_steps):
            observation = env.observation_space.sample()
            data.append(observation)
        env.close()
        return data

    def _train(self, participant_event: multiprocessing.Event, profiler_events: typing.List[multiprocessing.Event],
               gin_config_str: str, metric_values_dir: str):

        # Parse original gin config before parsing the participant's gin config
        gin.parse_config(self.gin_config_str)
        gin.parse_config(gin_config_str)
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
                @codecarbon.track_emissions(output_dir=metric_values_dir, output_file='train_emissions.csv', )
                def train():
                    return participant_module.train()

                train()
            else:
                participant_module.train()

        participant_event.clear()
        for profiler_event in profiler_events:
            profiler_event.clear()

    def perform_rollouts(self, participant_module, participant_model, env):
        rollout_rewards = []
        for _ in range(self.num_inference_episodes):
            observation = env.reset()
            done = False
            rewards = 0
            while not done:
                preprocessed_obs = participant_module.preprocess_observation(observation)
                action = participant_module.infer_once(model=participant_model, observation=preprocessed_obs)
                observation, reward, done, _ = env.step(action)
                rewards += reward
            rollout_rewards.append(rewards)
            print(f'Episode reward: {rewards}')
        return rollout_rewards

    def _inference(self,
                   participant_event: multiprocessing.Event,
                   profiler_events: typing.List[multiprocessing.Event],
                   inference_data: typing.List[typing.Any],
                   metrics_queue: multiprocessing.Queue,
                   metric_values_dir: str,
                   domain_config: str):

        metric_results = collections.defaultdict(dict)
        gin.parse_config(domain_config)
        participant_event.set()

        with working_directory(self.participant_module_path):
            participant_module, participant_module_spec = self._load_participant_module('inference.py')
            print(self.participant_module_path)
            print(participant_module_spec)
            participant_module_spec.loader.exec_module(participant_module)

            env = self.create_domain()
            participant_model = participant_module.load_model(env=env)

            preprocessed_data = [participant_module.preprocess_observation(x) for x in inference_data]

            def inference_step():
                return participant_module.infer_once(model=participant_model, observation=preprocessed_data[i])

            if self.time_inference_steps:
                inference_times = []
                for i in range(self.num_inference_steps):
                    inference_times.append(timeit.timeit(inference_step, number=1))

                metric_results['inference_time'] = dict(values=inference_times,
                                                        mean=np.mean(inference_times),
                                                        std=np.std(inference_times))

            if self.measure_emissions:
                @codecarbon.track_emissions(output_dir=metric_values_dir, output_file='inference_emissions.csv', )
                def perform_rollouts_and_track_emissions(p_module, p_model, e):
                    rewards = self.perform_rollouts(participant_module=p_module, participant_model=p_model, env=e)
                    return rewards

                logging.info('Performing rollouts and tracking emissions')
                all_rewards = perform_rollouts_and_track_emissions(p_module=participant_module,
                                                                   p_model=participant_model, e=env)
            else:
                logging.info('Performing rollouts')
                all_rewards = self.perform_rollouts(participant_module=participant_module,
                                                    participant_model=participant_model, env=env)

            logging.info(f'Rollout rewards: {all_rewards}')
            metric_results['rollout_returns'] = dict(values=all_rewards,
                                                     mean=np.mean(all_rewards),
                                                     std=np.std(all_rewards))
            metrics_queue.put(metric_results)

        participant_event.clear()
        for profiler_event in profiler_events:
            profiler_event.clear()

    def _run_inference_reliability_metrics(self, values, metric_values_dir):
        metrics = []
        for metric in self.reliability_metrics:
            if metric == ReliabilityMetrics.IqrAcrossRollouts:
                metrics.append(IqrAcrossRollouts())
            elif metric == ReliabilityMetrics.LowerCVaRAcrossRollouts:
                metrics.append(LowerCVaRAcrossRollouts())
            else:
                raise ValueError(f'Invalid metric: {metric}')

        with open(os.path.join(metric_values_dir, 'rollouts.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode_num', 'reward'])
            for i, value in enumerate(values):
                writer.writerow([i, value])

        evaluator = Evaluator(metrics=metrics, )
        reliability_metrics = evaluator.evaluate(
            run_paths=[os.path.join(metric_values_dir, 'rollouts.csv')], )
        return reliability_metrics

    def _run_train_reliability_metrics(self, save_dir):
        metrics = []
        for metric in self.reliability_metrics:
            if metric == ReliabilityMetrics.IqrWithinRuns:
                metrics.append(IqrWithinRuns())
            elif metric == ReliabilityMetrics.IqrAcrossRuns:
                metrics.append(IqrAcrossRuns())
            elif metric == ReliabilityMetrics.LowerCVaROnDiffs:
                metrics.append(LowerCVaROnDiffs())
            elif metric == ReliabilityMetrics.UpperCVaROnDrawdown:
                metrics.append(UpperCVaROnDrawdown())
            elif metric == ReliabilityMetrics.LowerCVarOnAcross:
                metrics.append(LowerCVaROnAcross())
            else:
                raise ValueError(f'Invalid metric: {metric}')

        logging.info(f'Running reliability metrics: {metrics}')
        logging.info(f'Logging to {save_dir}')

        if self.train_logs_dirs:
            logging.info(f'Searching for runs in {self.train_logs_dirs}')
            run_paths = []
            for pattern in self.train_logs_dirs:
                logging.info(f'Searching for runs in {pattern}')
                event_files = glob.glob(pattern, recursive=True)
                run_paths.extend(
                    [os.path.dirname(event_file) for event_file in event_files])

            logging.info(f'Found {len(run_paths)} runs in {self.train_logs_dirs}')
            logging.info(f'Run paths: {run_paths}')
            evaluator = Evaluator(metrics=metrics, )
            reliability_metrics = evaluator.evaluate(run_paths=run_paths, )
        else:
            logging.warning(f'No runs found in {self.train_logs_dirs}')
            reliability_metrics = {}
        self.metrics_results.update(reliability_metrics)
        return reliability_metrics

    def _run_training_benchmark(self):
        print(self.domain_config_paths)
        metric_results = {}
        metric_values_dir = os.path.join(self.metric_values_dir)
        if not os.path.exists(metric_values_dir):
            os.makedirs(metric_values_dir)

        if not self.run_offline_metrics_only:
            gin_config_str = gin.config_str()  # save gin configs for multiprocessing

            # Need a participant event to signal to profilers
            participant_started_event = multiprocessing.Event()
            profilers_started_events = [multiprocessing.Event() for _ in self.profilers]

            participant_process = self.mp_context.Process(target=self._train,
                                                          args=(participant_started_event, profilers_started_events,
                                                                gin_config_str, metric_values_dir,))
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

            participant_process.kill()
            logging.info('Killed participant process')

            for profiler in profilers:
                profiler.join()
                if profiler.is_alive():
                    logging.error(f'Profiler process {profiler.pid} is still running')
                elif profiler.exitcode != 0:
                    logging.error(f'Profiler process {profiler.pid} exited with code {profiler.exitcode}')
                else:
                    logging.info(f'Profiler process {profiler.pid} finished')

                profiler.kill()

        emissions_path = os.path.join(metric_values_dir, 'train_emissions.csv')
        assert os.path.exists(emissions_path), f'No train_emissions.csv found in {metric_values_dir}'

        seeds = [37, 82, 14]
        # replace the seed param in emissions file with the seeds to get different emissions file
        e_files = []
        for seed in seeds:
            e_files.append(emissions_path.replace('37', str(seed)))

        system_metrics = collect_system_metrics(emissions_file_paths=e_files,
                                                benchmark_mode=BenchmarkMode.TRAIN,
                                                metric_results=None, )

        logging.info(f'Collected system metrics')
        metric_results.update(system_metrics)
        logging.info(f'Updating metric results')
        if self.reliability_metrics:
            reliability_metrics = self._run_train_reliability_metrics(save_dir=metric_values_dir)

            for metric_name, metric in reliability_metrics.items():
                if metric_name == 'IqrWithinRuns':
                    metric['values'] = np.concatenate(metric['values']).tolist()
                metric_results[metric_name] = dict(values=metric['values'],
                                                   mean=np.mean(metric['values']),
                                                   std=np.std(metric['values']), )

        self.metrics_results.update(metric_results)

        with open(os.path.join(self.metric_values_dir, 'metric_results.json'), 'w') as f:
            print(self.metrics_results)
            json.dump(self.metrics_results, f)
        # pretty print all metrics with mean \pm std and units only if units are present
        # but we also want to use scientific notation for numbers that are too small
        for metric_name, metric in self.metrics_results.items():

            if 'mean' in metric.keys():
                # Determine if the number is small enough to use scientific notation
                # Here we use 1e-3 as the threshold, but you can adjust it as per your needs
                mean_format = '.2E' if abs(metric["mean"]) < SCI_NOTATION_THRESHOLD else '.2f'
                std_format = '.2E' if abs(metric["std"]) < SCI_NOTATION_THRESHOLD else '.2f'

                if 'units' in metric.keys():
                    print(
                        f'{metric_name}: {metric["mean"]:{mean_format}} \pm {metric["std"]:{std_format}} {metric["units"]}')
                else:
                    print(f'{metric_name}: {metric["mean"]:{mean_format}} \pm {metric["std"]:{std_format}}')

        # Plot metrics and save to file
        if self.plot_metrics:
            for profiler_object in self.profilers:
                title, fig = profiler_object.plot_results()
                plt.savefig(os.path.join(self.metric_values_dir, f'{title}.png'))
            self._plot_metrics(metric_results=self.metrics_results, save_dir=self.metric_values_dir)

    def _run_inference_benchmark(self):
        logging.info(f'Running inference benchmark for domain: {self.domain}')
        if not self.run_offline_metrics_only:
            for domain_config_path in self.domain_config_paths:
                logging.info(f'Running inference benchmark for domain config: {domain_config_path}')

                metric_results = {}
                domain_config_name = os.path.splitext(os.path.basename(domain_config_path))[0]
                metric_values_dir = os.path.join(self.metric_values_dir, domain_config_name)
                if not os.path.exists(metric_values_dir):
                    os.makedirs(metric_values_dir)

                # Parsing gin config so that "create_domain" can receive different arguments
                gin.parse_config_file(domain_config_path)
                gin_config_str = gin.config_str()  # save gin configs for multiprocessing

                inference_data = self._get_observation_data()

                # Need a participant event to signal to profilers
                participant_started_event = multiprocessing.Event()
                profilers_started_events = [multiprocessing.Event() for _ in self.profilers]

                metrics_data_queue = self.mp_context.Queue()
                participant_process = self.mp_context.Process(target=self._inference,
                                                              args=(participant_started_event,
                                                                    profilers_started_events,
                                                                    inference_data,
                                                                    metrics_data_queue,
                                                                    metric_values_dir,
                                                                    gin_config_str))

                participant_process.start()
                profilers = _start_profilers(profilers=self.profilers,
                                             participant_event=participant_started_event,
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

                participant_process.kill()
                logging.info('Killed participant process')

                for profiler in profilers:
                    profiler.join()
                    if profiler.is_alive():
                        logging.error(f'Profiler process {profiler.pid} is still running')
                    elif profiler.exitcode != 0:
                        logging.error(f'Profiler process {profiler.pid} exited with code {profiler.exitcode}')
                    else:
                        logging.info(f'Profiler process {profiler.pid} finished')

                logging.info(f'Waiting for inference metrics to be collected...')
                inference_metrics = metrics_data_queue.get()
                logging.info(f'Collected inference metrics: {inference_metrics}')

                metric_results.update(inference_metrics)
                emissions_path = os.path.join(metric_values_dir, 'inference_emissions.csv')
                assert os.path.exists(emissions_path), f'No inference_emissions.csv found in {metric_values_dir}'
                # system_metrics = self.collect_system_metrics(emissions_file_path=emissions_path,
                #                                              benchmark_mode=BenchmarkMode.INFERENCE,
                #                                              metric_results=metric_results,
                #                                              )
                # logging.info(f'Collected system metrics: {system_metrics}')
                # metric_results.update(system_metrics)
                if self.reliability_metrics:
                    reliability_metrics = self._run_inference_reliability_metrics(
                        values=metric_results['rollout_returns']['values'],
                        metric_values_dir=metric_values_dir)
                    metric_results.update(reliability_metrics)
                self.metrics_results[domain_config_name] = metric_results

                with open(os.path.join(metric_values_dir, 'inference_metric_results.json'), 'w') as f:
                    json.dump(self.metrics_results, f)

    def _plot_metrics(self, save_dir, metric_results):
        if not self.metrics_results:
            logging.warning('No metrics to plot')

        for metric_name, metric_values in metric_results.items():
            plt.figure()
            plt.title(metric_name)
            plt.plot(metric_values['values'])
            plt.savefig(os.path.join(save_dir, f'{metric_name}.png'))

    def run_benchmark(self):
        self.gin_config_str = gin.config_str()  # save gin configs for multiprocessing
        if self.mode == BenchmarkMode.TRAIN:
            self._run_training_benchmark()
        elif self.mode == BenchmarkMode.INFERENCE:
            self._run_inference_benchmark()
        else:
            raise ValueError('Benchmark mode must be either train or inference')
