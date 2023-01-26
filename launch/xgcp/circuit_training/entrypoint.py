import os
import platform
import socket
import subprocess

from absl import app
from absl import flags
from absl import logging


flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_integer('global_seed', 0, 'Global seed.')
flags.DEFINE_integer('reverb_port', 0, 'Reverb port.')
flags.DEFINE_string('reverb_server_ip', None, 'Reverb server.')
flags.DEFINE_string('netlist_file', None, 'Netlist file.')
flags.DEFINE_string('init_placement', None, 'Initial placement.')
flags.DEFINE_integer('num_collect_jobs', 0, 'Number of collect jobs.')
flags.DEFINE_string('python_version', '', 'Python version.')
flags.DEFINE_string('gin_config', None, 'Gin config file.')
flags.DEFINE_string('participant_module_path', None, 'Participant module path.')
flags.DEFINE_boolean(
    'run_offline_metrics_only', False, 'Run offline metrics only.'
)

flags.DEFINE_enum(
    'job_type',
    default=None,
    enum_values=['local', 'collect', 'reverb', 'eval', 'train'],
    help='Type of the job',
)

_DEBUG = flags.DEFINE_boolean('debug', False, 'run in debug mode.')


FLAGS = flags.FLAGS


def main(_):
  # set environment variables
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
  os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
  os.environ['ROOT_DIR'] = FLAGS.root_dir
  os.environ['GLOBAL_SEED'] = str(FLAGS.global_seed)
  os.environ['REVERB_PORT'] = str(FLAGS.reverb_port)
  os.environ['REVERB_SERVER_IP'] = FLAGS.reverb_server_ip
  os.environ['NETLIST_FILE'] = FLAGS.netlist_file
  os.environ['INIT_PLACEMENT'] = FLAGS.init_placement
  os.environ['NUM_COLLECT_JOBS'] = str(FLAGS.num_collect_jobs)
  os.environ['JOB_TYPE'] = FLAGS.job_type

  # Get the current working directory
  cwd = os.getcwd()

  # Add the current working directory to the PYTHONPATH environment variable
  os.environ['PYTHONPATH'] = cwd + os.pathsep + os.getenv('PYTHONPATH', '')

  os.makedirs(FLAGS.root_dir, exist_ok=True)

  # Need different metric values dirs since we're running multiple main_submissions
  hostname = socket.gethostname()
  metric_values_dir = os.path.join(
      FLAGS.root_dir,
      'metrics',
      hostname,
  )

  command = (
      f'{FLAGS.python_version} rl_perf/submission/main_submission.py '
      f' --gin_config={FLAGS.gin_config} '
      f' --participant_module_path={FLAGS.participant_module_path} '
      f' --root_dir={FLAGS.root_dir} '
      f' --metric_values_dir={metric_values_dir} '
      f' --train_logs_dirs={FLAGS.root_dir} '
      f' --run_offline_metrics_only={FLAGS.run_offline_metrics_only}'
  )

  process = subprocess.Popen(command, shell=True)
  process.communicate()


if __name__ == '__main__':
  app.run(main)
