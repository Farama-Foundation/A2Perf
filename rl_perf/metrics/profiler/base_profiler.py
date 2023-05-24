import multiprocessing
import typing


class BaseProfiler(object):

    def __init__(self, participant_process_event: multiprocessing.Event = None,
                 participant_process: multiprocessing.Process = None,
                 profiler_event: multiprocessing.Event = None,
                 base_log_dir: typing.Optional[str] = None):
        self.participant_process_event = participant_process_event
        self.profiler_event = profiler_event
        self.participant_process = participant_process
        self.base_log_dir = base_log_dir

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def get_metric_results(self):
        raise NotImplementedError
