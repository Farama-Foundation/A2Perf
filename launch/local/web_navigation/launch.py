import os
import subprocess

from absl import app
from absl import flags

# Updated to define a list of motion files
flags.DEFINE_list('difficulty_levels', None,
                  'List of motion file names without extensions')
flags.DEFINE_string('algo', None, 'Algorithm to use')
flags.DEFINE_list('seed', None, 'List of seed numbers')
flags.DEFINE_string('experiment_number', None, 'Experiment number')
flags.DEFINE_boolean('debug', False, 'Enable debug mode')
flags.DEFINE_string('host_dir_base',

                    None, 'Base directory for host')

FLAGS = flags.FLAGS


def get_next_experiment_number(host_dir_base):
  try:
    base_number = host_dir_base.rstrip('/')
    print(base_number)
    last_exp_num = \
      sorted([int(d) for d in os.listdir(host_dir_base) if d.isdigit()])[-1]
  except IndexError:
    return "0001"
  except FileNotFoundError:
    return "0001"
  return f"{last_exp_num + 1:04d}"


def main(_):
  # change directory to the root of the repo
  os.chdir(
      os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../')
  )

  debug_path = "debug" if FLAGS.debug else ""
  host_dir_base = os.path.abspath(FLAGS.host_dir_base)

  # Docker cleanup, but don't fail if it doesn't exist
  subprocess.run(["docker", "rm", "-f", "web_nav_container"],
                 check=False)

  seeds = [int(seed) for seed in FLAGS.seed]
  base_gin_config = f'/rl-perf/rl_perf/submission/configs/web_navigation/' + debug_path

  for difficulty_level in FLAGS.difficulty_levels:  # Loop through each motion file
    # host_dir_base = f"/home/ikechukwuu/workspace/gcs/a2perf/web_navigation/difficulty_level_{difficulty_level}/{FLAGS.algo}/{debug_path}"
    host_dir_base = f'{host_dir_base}/difficulty_level_{difficulty_level}/{FLAGS.algo}/{debug_path}'
    root_dir_base = f"/mnt/gcs/a2perf/web_navigation/difficulty_level_{difficulty_level}/{FLAGS.algo}/{debug_path}"

    for seed in seeds:
      next_exp_num = get_next_experiment_number(host_dir_base)
      next_exp_num = FLAGS.experiment_number if FLAGS.experiment_number is not None else next_exp_num
      # Launch the xmanager command
      xmanager_cmd = [
          f'env/bin/xmanager', 'launch',
          # "/home/ikechukwuu/workspace/rl-perf/env/bin/xmanager", "launch",
          "launch/xm_launch_web_navigation.py", "--",
          f"--root_dir={root_dir_base.rstrip('/')}/{next_exp_num}",
          f"--participant_module_path={os.path.join('/rl-perf/rl_perf/rlperf_benchmark_submission/web_navigation', FLAGS.algo, debug_path)}",
          f"--difficulty_level={difficulty_level}",
          f"--gin_config={os.path.join(base_gin_config, 'train.gin')}",
          "--local",
          f"--algo={FLAGS.algo}",
          '--debug' if FLAGS.debug else '',
          f"--seed={seed}",
          f"--experiment_number={next_exp_num}"
      ]

      subprocess.run(xmanager_cmd, check=True)


if __name__ == "__main__":
  app.run(main)
