import os
import subprocess

from absl import app
from absl import flags

flags.DEFINE_string('root_dir', None, 'Root directory.')
flags.DEFINE_multi_string(
    'train_logs_dirs',
    ['train_logs'],
    'Directories for train logs from all of the experiments that reliability'
    ' metrics will be calculated on',
)
flags.DEFINE_integer('seed', 0, 'Global seed.')
flags.DEFINE_integer('num_websites', 1, 'Number of websites.')
flags.DEFINE_integer(
    'env_batch_size', 8, 'Batch size for the environment.'
)  # added default
flags.DEFINE_integer(
    'total_env_steps', 10000, 'Total steps in the environment.'
)  # added default
flags.DEFINE_string(
    'difficulty_level', '0', 'Difficulty level of the experiment.'
)  # changed default to '0'
flags.DEFINE_string('gin_config', None, 'Gin config file.')
flags.DEFINE_string('participant_module_path', None, 'Participant module path.')
flags.DEFINE_boolean(
    'run_offline_metrics_only', False, 'Run offline metrics only.'
)
flags.DEFINE_integer(
    'rb_capacity', 50000, 'Replay buffer capacity.'
)  # added default
flags.DEFINE_integer(
    'batch_size', 64, 'Batch size for training.'
)  # added default
flags.DEFINE_integer(
    'eval_interval', 100, 'Evaluation interval.'
)  # added default
flags.DEFINE_integer(
    'train_checkpoint_interval', 100, 'Train checkpoint interval.'
)  # added default
flags.DEFINE_integer(
    'policy_checkpoint_interval', 1000, 'Policy checkpoint interval.'
)  # added default
flags.DEFINE_integer(
    'rb_checkpoint_interval', 20000, 'RB checkpoint interval.'
)  # added default
flags.DEFINE_integer('log_interval', 10000, 'Log interval.')  # added default
flags.DEFINE_integer(
    'summary_interval', 10000, 'Summary interval.'
)  # added default
flags.DEFINE_integer(
    'timesteps_per_actorbatch', 0, 'Summary interval.'
)  # added default
flags.DEFINE_float('learning_rate', 0, 'the learning rate')

FLAGS = flags.FLAGS


def main(_):
  # Set environment variables
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
  os.environ['SEED'] = str(FLAGS.seed)
  os.environ['ENV_BATCH_SIZE'] = str(FLAGS.env_batch_size)
  os.environ['TOTAL_ENV_STEPS'] = str(FLAGS.total_env_steps)
  os.environ['ROOT_DIR'] = FLAGS.root_dir
  os.environ['DIFFICULTY_LEVEL'] = FLAGS.difficulty_level
  os.environ['EVAL_INTERVAL'] = str(FLAGS.eval_interval)
  os.environ['TRAIN_CHECKPOINT_INTERVAL'] = str(FLAGS.train_checkpoint_interval)
  os.environ['POLICY_CHECKPOINT_INTERVAL'] = str(
      FLAGS.policy_checkpoint_interval
  )
  os.environ['NUM_WEBSITES'] = str(FLAGS.num_websites)
  os.environ['RB_CHECKPOINT_INTERVAL'] = str(FLAGS.rb_checkpoint_interval)
  os.environ['LOG_INTERVAL'] = str(FLAGS.log_interval)
  os.environ['SUMMARY_INTERVAL'] = str(FLAGS.summary_interval)
  os.environ['LEARNING_RATE'] = str(FLAGS.learning_rate)
  os.environ['TIMESTEPS_PER_ACTORBATCH'] = str(FLAGS.timesteps_per_actorbatch)
  # New environment variables for rb_capacity and batch_size
  os.environ['RB_CAPACITY'] = str(FLAGS.rb_capacity)
  os.environ['BATCH_SIZE'] = str(FLAGS.batch_size)

  # Get the current working directory
  cwd = os.getcwd()

  # Add the current working directory to the PYTHONPATH environment variable
  os.environ['PYTHONPATH'] = cwd + os.pathsep + os.getenv('PYTHONPATH', '')

  os.makedirs(FLAGS.root_dir, exist_ok=True)
  train_logs_dirs_str = ','.join(FLAGS.train_logs_dirs)

  command = (
      'python3.8 a2perf/submission/main_submission.py '
      f'--gin_config={FLAGS.gin_config} '
      f'--participant_module_path={FLAGS.participant_module_path} '
      f'--root_dir={FLAGS.root_dir} '
      f'--train_logs_dirs={train_logs_dirs_str} '
      f'--run_offline_metrics_only={FLAGS.run_offline_metrics_only}'
  )

  process = subprocess.Popen(command, shell=True)
  process.communicate()


if __name__ == '__main__':
  app.run(main)
