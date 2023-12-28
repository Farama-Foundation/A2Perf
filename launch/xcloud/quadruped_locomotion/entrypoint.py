import os
import subprocess

from absl import app
from absl import flags

# Define only the necessary flags
_DEBUG = flags.DEFINE_bool('debug', False, 'debugging mode')
flags.DEFINE_integer('seed', 0, 'Global seed.')
flags.DEFINE_integer('num_parallel_cores', 1, 'Number of parallel cores.')
flags.DEFINE_integer('num_epochs', 0, 'Number of epochs.')
flags.DEFINE_integer(
    'total_env_steps', 10000, 'Total steps in the environment.'
)
flags.DEFINE_string('motion_file_path', None, 'Motion file path.')
flags.DEFINE_string('algo', 'default_algo', 'Algorithm used.')

flags.DEFINE_multi_string(
    'train_logs_dirs',
    ['train_logs'],
    'Directories for train logs from all of the experiments that reliability'
    ' metrics will be calculated on',
)

flags.DEFINE_enum(
    'skill_level',
    None,
    ['novice', 'intermediate', 'expert'],
    'Skill level of the expert.',
)
flags.DEFINE_enum(
    'task', None, ['dog_pace', 'dog_trot', 'dog_spin'], 'Task to run.'
)
flags.DEFINE_string('gin_config', None, 'Gin config file.')
flags.DEFINE_string('mode', 'train', 'Mode of execution.')
flags.DEFINE_integer('int_save_freq', 1000, 'Interval save frequency.')
flags.DEFINE_integer('int_eval_freq', 1000, 'Interval evaluation frequency.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
flags.DEFINE_string('root_dir', '/tmp', 'Root directory.')
flags.DEFINE_string('participant_module_path', None, 'Participant module path.')
flags.DEFINE_boolean(
    'run_offline_metrics_only', False, 'Run offline metrics only.'
)
FLAGS = flags.FLAGS


def main(_):
  os.environ['OMPI_MCA_btl_vader_single_copy_mechanism'] = 'none'
  os.environ['SEED'] = str(FLAGS.seed)
  os.environ['PARALLEL_CORES'] = str(FLAGS.num_parallel_cores)
  os.environ['TOTAL_ENV_STEPS'] = str(FLAGS.total_env_steps)
  os.environ['INT_SAVE_FREQ'] = str(FLAGS.int_save_freq)
  os.environ['INT_EVAL_FREQ'] = str(FLAGS.int_eval_freq)
  os.environ['LEARNING_RATE'] = str(FLAGS.learning_rate)
  os.environ['BATCH_SIZE'] = str(FLAGS.batch_size)
  os.environ['ROOT_DIR'] = FLAGS.root_dir
  os.environ['PARALLEL_MODE'] = 'True'
  os.environ['MODE'] = FLAGS.mode
  os.environ['SKILL_LEVEL'] = FLAGS.skill_level
  os.environ['NUM_EPOCHS'] = str(FLAGS.num_epochs)
  os.environ['VISUALIZE'] = 'False'
  os.environ['DATASET_ID'] = (
      f'QuadrupedLocomotion-{FLAGS.task}-{FLAGS.skill_level}-v0'
  )
  os.environ['MINARI_DATASETS_PATH'] = '/workdir/a2perf/datasets/data'
  os.environ['MOTION_FILE_PATH'] = FLAGS.motion_file_path
  os.environ['SETUP_PATH'] = f'{FLAGS.algo}_actor.py'
  os.environ['TASK'] = FLAGS.task

  cwd = os.getcwd()
  os.environ['PYTHONPATH'] = cwd + os.pathsep + os.getenv('PYTHONPATH', '')
  os.makedirs(FLAGS.root_dir, exist_ok=True)
  train_logs_dirs_str = ','.join(FLAGS.train_logs_dirs)

  participant_module_path = FLAGS.participant_module_path
  command = (
      'python3.9 a2perf/submission/main_submission.py '
      f'--gin_config={FLAGS.gin_config} '
      f'--participant_module_path={participant_module_path} '
      f'--root_dir={FLAGS.root_dir} '
      f'--train_logs_dirs={train_logs_dirs_str} '
      f'--run_offline_metrics_only={FLAGS.run_offline_metrics_only}'
  )

  process = subprocess.Popen(command, shell=True, env=os.environ.copy())
  process.communicate()


if __name__ == '__main__':
  app.run(main)
