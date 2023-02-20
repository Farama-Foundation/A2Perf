class BaseProfiler(object):
    def __init__(self, ):
        self.participant_process = None

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def set_participant_process(self, process):
        self.participant_process = process
