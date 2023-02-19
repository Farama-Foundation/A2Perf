import gin
import logging
import psutil
import subprocess


@gin.configurable
class Submission:
    def __init__(self, participant_module, participant_module_path, participant_module_spec, benchmark, profilers=None,
                 mode='training',
                 domain='circuit_training',
                 model_location=None):

        self.profilers = profilers if profilers is not None else []
        if mode not in ['train', 'eval', 'infer']:
            raise ValueError('Mode must be one of train, eval, or infer')

        if domain not in ['circuit_training', 'web_navigation', 'quadruped_locomotion']:
            raise ValueError('Domain must be circuit_training, web_navigation, or quadruped_locomotion')

        self.participant_module = participant_module
        self.participant_module_spec = participant_module_spec
        self.participant_module_path = participant_module_path
        self.domain = domain
        self.benchmark = benchmark
        self.mode = mode
        self.model_location = model_location

    def run_benchmark(self):
        # Run the participant's module in a subprocess
        cmd = f"python3 -c \"import importlib; participant_module_spec = importlib.util.spec_from_file_location(" \
              f"'train', '{self.participant_module_path}'); participant_module = importlib.util.module_from_spec(" \
              f"participant_module_spec); participant_module_spec.loader.exec_module(participant_module); " \
              f"participant_module.train()\""
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, err = p.communicate()
        if err:
            logging.error(f"Error running participant module: {err}")
            return

        # Get the process ID of the subprocess
        pid = p.pid
        logging.info(f"Participant module process ID: {pid}")

        # Pass the subprocess to the profilers
        for profiler in self.profilers:
            profiler.set_pid(pid)

        # Wait for the subprocess to finish
        p.wait()
        logging.info(f"Participant module process {pid} finished")
