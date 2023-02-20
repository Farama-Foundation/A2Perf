import importlib
import logging
import multiprocessing
from enum import Enum
from multiprocessing import Process, Queue

import gin


@gin.configurable
class BenchmarkMode(Enum):
    TRAIN = 'train'
    EVAL = 'eval'


@gin.configurable
class BenchmarkDomain(Enum):
    WEB_NAVIGATION = 'web_navigation'
    CIRCUIT_TRAINING = 'circuit_training'
    QUADRUPED_LOCOMOTION = 'quadruped_locomotion'


def _start_profilers(profilers, process):
    processes = []
    for profiler_class in profilers:
        logging.info(f'Starting profiler: {profiler_class}')
        profiler = profiler_class()
        profiler.set_participant_process(process=process)
        process = multiprocessing.Process(target=profiler.start)
        process.start()
        processes.append(process)
    return processes


@gin.configurable
class Submission:
    def __init__(self,
                 participant_module_path,
                 profilers=None,
                 mode=BenchmarkMode.TRAIN,
                 domain=BenchmarkDomain.WEB_NAVIGATION):
        self.profilers = profilers if profilers is not None else []
        self.participant_module_path = participant_module_path
        self.domain = domain
        self.mode = mode

    def _train(self, queue=None):
        participant_module_spec = importlib.util.spec_from_file_location("train", self.participant_module_path)
        participant_module = importlib.util.module_from_spec(participant_module_spec)
        participant_module_spec.loader.exec_module(participant_module)
        queue.put(None)

    def _eval(self):
        pass

    def run_benchmark(self):
        q = Queue()
        if self.mode == 'train':
            participant_process = Process(target=self._train, args=(q,))
        elif self.mode == 'eval':
            participant_process = Process(target=self._eval, args=(self.participant_module_path, q))
        elif self.mode == 'infer':
            participant_process = Process(target=self._infer, args=(self.participant_module_path, q))
        else:
            raise ValueError('Mode must be one of train, eval, or infer')

        _start_profilers(self.profilers, participant_process)

        logging.info(f'Participant module process ID: {participant_process.pid}')
        participant_process.start()
        q.get()
        participant_process.join()
        logging.info(f'Participant module process {participant_process.pid} finished')
