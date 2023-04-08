import enum
import os
import importlib
import logging
import multiprocessing
import typing

import gin

import gym

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
                 reliability_metrics: typing.List[ReliabilityMetrics] = None):

        self.base_log_dir = base_log_dir
        if self.base_log_dir is not None:
            os.makedirs(self.base_log_dir, exist_ok=True)

        self.metric_values_dir = metric_values_dir
        if self.metric_values_dir is not None:
            os.makedirs(self.metric_values_dir, exist_ok=True)
        self.num_inference_steps = num_inference_steps
        self.profilers = profilers if profilers is not None else []
        self.participant_module_path = participant_module_path
        self.domain = domain
        self.mode = mode
        self.reliability_metrics = reliability_metrics

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

        # Generate fake data for inference using a tensorspec. To avoid clashing with the participant code,
        # we will generate all of the tensorspecs beforehand.
        env = gym.make(self.domain.value, difficulty=1, seed=0)
        data = []
        for _ in range(self.num_inference_steps):
            observation = env.observation_space.sample()
            preprocessed_obs = participant_module.preprocess_observation(observation)
            data.append(preprocessed_obs)

        # Get the participant model
        participant_model = participant_module.load_model()

        # Since all the startup is done, we can communicate to the profilers that they can start profiling
        participant_event.set()

        participants_actions = []
        for i in range(self.num_inference_steps):
            action = participant_module.infer_once(model=participant_model, observation=data[i])
            participants_actions.append(action)
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
        evaluator.evaluate(run_paths=run_paths, outfile_prefix=self.metric_values_dir)

    def run_benchmark(self):

        # Need a participant event to signal to profilers
        participant_started_event = multiprocessing.Event()
        profilers_started_events = [multiprocessing.Event() for _ in self.profilers]
        if self.mode == BenchmarkMode.TRAIN:
            participant_process = multiprocessing.Process(target=self._train,
                                                          args=(participant_started_event, profilers_started_events))
        elif self.mode == BenchmarkMode.INFERENCE:
            participant_process = multiprocessing.Process(target=self._infer,
                                                          args=(participant_started_event, profilers_started_events))
        else:
            raise ValueError('Benchmark mode must be either train or infer')

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

        if self.reliability_metrics:
            self.run_reliability_metrics()
