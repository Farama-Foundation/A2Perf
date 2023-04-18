import multiprocessing
import os.path
import time
import typing

import gin
import matplotlib.pyplot as plt
import psutil
from absl import logging

from rl_perf.metrics.profiler.inference_profiler import InferenceProfiler
from rl_perf.metrics.profiler.training_profiler import TrainingProfiler


def compute_memory(process):
    try:
        children = process.children(recursive=True)
        children.append(process)
        rss = 0
        vms = 0
        shared = 0
        for child in children:
            mem_info = child.memory_info()
            rss += mem_info.rss
            vms += mem_info.vms
            shared += mem_info.shared
        return rss, vms, shared
    except psutil.NoSuchProcess:
        return 0, 0, 0


@gin.configurable
class InferenceMemoryProfiler(InferenceProfiler):
    def __init__(self, participant_process_event: multiprocessing.Event, profiler_event: multiprocessing.Event,
                 base_log_dir: str, interval: int = 1,
                 pipe_for_participant_process: typing.Optional[multiprocessing.Pipe] = None,
                 participant_process: typing.Optional[multiprocessing.Process] = None,
                 log_file: str = 'inference_memory_profiler.csv'):
        super().__init__(participant_process_event=participant_process_event, participant_process=participant_process,
                         base_log_dir=base_log_dir, profiler_event=profiler_event)

        self.pipe_for_participant_process = pipe_for_participant_process
        self.interval = interval
        self.participant_ps_process = None
        self.log_file_path = os.path.join(self.base_log_dir, log_file)
        self.values = []

    def start(self):
        logging.info('Starting inference memory profiler')
        while not self.participant_process_event.is_set():
            time.sleep(1)
            logging.info('Inference Memory Profiler: Waiting for participant process to start')
        logging.info('Inference Memory Profiler: Participant process started')

        participant_process_pid = self.pipe_for_participant_process.recv()
        self.participant_ps_process = psutil.Process(participant_process_pid)
        logging.info(f'Inference Memory Profiler: Participant process PID: {self.participant_ps_process.pid}')

        self.profiler_event.set()

        with open(self.log_file_path, 'w') as self.output_file:
            while self.participant_process_event.is_set():
                time.sleep(self.interval)
                logging.info('Inference Memory Profiler: Profiling memory')
                self._profile_memory()
            else:
                logging.warning('Inference Memory Profiler: Participant process stopped')

    def _profile_memory(self):
        rss, vms, shared = compute_memory(self.participant_ps_process)
        rss, vms, shared = psutil._common.bytes2human(rss), psutil._common.bytes2human(vms), psutil._common.bytes2human(
                shared)
        self.output_file.write(f'{time.time()},{rss},{vms},{shared}\n')
        self.values.append((time.time(), rss, vms, shared))
        logging.info(f'Memory Profiler: RSS: {rss}, VMS: {vms}, Shared: {shared}')

    def get_metric_results(self):
        timestamps, rss, vms, shared = zip(*self.values)
        result = dict(
                rss=rss,
                vms=vms,
                shared=shared,
                timestamps=timestamps
                )

        return result

    @gin.configurable('InferenceMemoryProfiler.plot_results')
    def plot_results(self, **kwargs):
        timestamps, rss, vms, shared = zip(*self.values)
        plt.plot(timestamps, rss, label='RSS')
        plt.plot(timestamps, vms, label='VMS')
        plt.plot(timestamps, shared, label='Shared')
        plt.legend()
        plt.savefig(save_path)


@gin.configurable
class TrainingMemoryProfiler(TrainingProfiler):

    def __init__(self, participant_process_event, profiler_event, participant_process, base_log_dir, interval=1,
                 log_file='memory_profiler.csv'):
        """Profiles memory usage of the participant process with psutil:
        https://psutil.readthedocs.io/en/latest/index.html?highlight=Process#process-class.

        Args:
            participant_process_event: Event that is set when the participant process starts.
            participant_process: The participant process.
            interval: The interval in seconds to check memory usage.
        """
        super().__init__(participant_process_event=participant_process_event, participant_process=participant_process,
                         base_log_dir=base_log_dir, profiler_event=profiler_event)
        self.interval = interval
        self.participant_ps_process = None
        self.log_file_path = os.path.join(self.base_log_dir, log_file)

    def start(self):
        logging.info('Starting memory profiler')
        while not self.participant_process_event.is_set():
            time.sleep(1)
            logging.info('Memory Profiler: Waiting for participant process to start')
        logging.info('Memory Profiler: Participant process started')

        self.profiler_event.set()
        with open(self.log_file_path, 'w') as self.output_file:
            while self.participant_process_event.is_set():
                time.sleep(self.interval)
                logging.info('Memory Profiler: Profiling memory')
                self._profile_memory()
            else:
                logging.warning('Memory Profiler: Participant process stopped')

    def _profile_memory(self):
        if self.participant_ps_process is None:
            self.participant_ps_process = psutil.Process(self.participant_process.pid)
        logging.info(f'Memory Profiler: Participant process PID: {self.participant_ps_process.pid}')

        rss, vms, shared = compute_memory(self.participant_ps_process)
        rss, vms, shared = psutil._common.bytes2human(rss), psutil._common.bytes2human(vms), psutil._common.bytes2human(
                shared)
        self.output_file.write(f'{time.time()},{rss},{vms},{shared}\n')
        logging.info(f'Memory Profiler: RSS: {rss}, VMS: {vms}, Shared: {shared}')

    def stop(self):
        pass
