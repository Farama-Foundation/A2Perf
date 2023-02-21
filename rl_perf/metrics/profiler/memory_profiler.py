import logging
import os.path
import time
import psutil
import gin

from rl_perf.metrics.profiler.eval_profiler import EvaluationProfiler
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


class EvaluationMemoryProfiler(EvaluationProfiler):
    def __init__(self):
        super(EvaluationMemoryProfiler, self).__init__()
