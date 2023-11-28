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
  os.chdir("/home/ikechukwuu/workspace/rl-perf")

  debug_path = "debug" if FLAGS.debug else ""

  seeds = [int(seed) for seed in FLAGS.seed]
  base_gin_config = f'/rl-perf/rl_perf/submission/configs/web_nav/' + debug_path

  for difficulty_level in FLAGS.difficulty_levels:  # Loop through each motion file
    host_dir_base = f"/home/ikechukwuu/workspace/gcs/a2perf/web_nav/difficulty_level_{difficulty_level}/{FLAGS.algo}/{debug_path}"
    root_dir_base = f"/mnt/gcs/a2perf/web_nav/difficulty_level_{difficulty_level}/{FLAGS.algo}/{debug_path}"

    for seed in seeds:
      next_exp_num = get_next_experiment_number(host_dir_base)
      next_exp_num = FLAGS.experiment_number if FLAGS.experiment_number is not None else next_exp_num
      # Launch the xmanager command
      xmanager_cmd = [
          "/home/ikechukwuu/workspace/rl-perf/env/bin/xmanager", "launch",
          "launch/xm_launch_web_nav.py", "--",
          f"--root_dir={root_dir_base.rstrip('/')}/{next_exp_num}",
          f"--participant_module_path={os.path.join('/rl-perf/rl_perf/rlperf_benchmark_submission/web_nav', FLAGS.algo, debug_path)}",
          f"--difficulty_level={difficulty_level}",
          f"--gin_config={os.path.join(base_gin_config, 'train.gin')}",
          "--local",
          f"--algo={FLAGS.algo}",
          '--debug' if FLAGS.debug else '',
          f"--seed={seed}",
          f"--experiment_number={next_exp_num}"
      ]

      subprocess.run(xmanager_cmd, check=True)

      # Docker cleanup
      subprocess.run(["docker", "rm", "-f", "web_nav_container"],
                     check=True)


if __name__ == "__main__":
  app.run(main)