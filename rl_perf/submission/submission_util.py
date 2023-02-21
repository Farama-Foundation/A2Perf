import enum
import os
import importlib
import logging
import multiprocessing
import typing

import gin

from metrics.profiler.base_profiler import BaseProfiler


@gin.configurable
class BenchmarkMode(enum.Enum):
    TRAIN = 'train'
    EVAL = 'eval'


@gin.configurable
class BenchmarkDomain(enum.Enum):
    WEB_NAVIGATION = 'web_navigation'
    CIRCUIT_TRAINING = 'circuit_training'
    QUADRUPED_LOCOMOTION = 'quadruped_locomotion'


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
                 participant_module_path,
                 profilers=None,
                 mode=BenchmarkMode.TRAIN,
                 domain=BenchmarkDomain.WEB_NAVIGATION,
                 base_log_dir=None):

        self.base_log_dir = base_log_dir
        if self.base_log_dir is not None:
            os.makedirs(self.base_log_dir, exist_ok=True)

        self.profilers = profilers if profilers is not None else []
        self.participant_module_path = participant_module_path
        self.domain = domain
        self.mode = mode

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

    def _eval(self):
        pass

    def run_benchmark(self):
        participant_started_event = multiprocessing.Event()
        profilers_started_events = [multiprocessing.Event() for _ in self.profilers]
        if self.mode == 'train':
            participant_process = multiprocessing.Process(target=self._train,
                                                          args=(participant_started_event, profilers_started_events))
        elif self.mode == 'eval':
            participant_process = multiprocessing.Process(target=self._eval, args=(self.participant_module_path, q))
        elif self.mode == 'infer':
            participant_process = multiprocessing.Process(target=self._infer, args=(self.participant_module_path, q))
        else:
            raise ValueError('Mode must be one of train, eval, or infer')

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
