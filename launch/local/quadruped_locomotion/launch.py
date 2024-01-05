import os
import subprocess

from absl import app
from absl import flags

FLAGS = flags.FLAGS
# Define flags
_MOTION_FILES = flags.DEFINE_list(
    'motion_files', None, 'List of motion file names without extensions')
_ALGO = flags.DEFINE_list('algo', None, 'List of algorithms')
_SEED = flags.DEFINE_list('seed', None, 'List of seed numbers')
_EXPERIMENT_NUMBER = flags.DEFINE_string(
    'experiment_number', None, 'Experiment number')
_DEBUG = flags.DEFINE_boolean('debug', False, 'Enable debug mode')
_MODE = flags.DEFINE_string('mode', None, 'Mode to run in')
_SKILL_LEVEL = flags.DEFINE_string('skill_level', None, 'Skill level')
_HOST_DIR_BASE = flags.DEFINE_string('host_dir_base',

                                     None, 'Base directory for host')


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
  host_dir_base = os.path.expanduser(_HOST_DIR_BASE.value)
  debug_path = "debug" if _DEBUG.value else ""
  seeds = [int(seed) for seed in _SEED.value]
  base_gin_config = f'/rl-perf/a2perf/submission/configs/quadruped_locomotion/' + debug_path
  gin_config_name = f'train.gin' if _MODE.value == 'train' else f'inference.gin'
  env_mode = 'train' if _MODE.value == 'train' else 'test'

  for algo in _ALGO.value:  # Loop through each algorithm
    for motion_file in _MOTION_FILES.value:  # Loop through each motion file
      host_dir_base = f"{host_dir_base}/gcs/a2perf/quadruped_locomotion/{motion_file}/{algo}/{debug_path}"
      root_dir_base = f"/mnt/gcs/a2perf/quadruped_locomotion/{motion_file}/{algo}/{debug_path}"

      for seed in seeds:
        subprocess.run(
            ["docker", "rm", "-f", "quadruped_locomotion_container"],
            check=True)

        next_exp_num = get_next_experiment_number(host_dir_base)

        # Construct the motion file path
        motion_file_path = f"/rl-perf/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/{motion_file}.txt"

        next_exp_num = FLAGS.experiment_number if FLAGS.experiment_number is not None else next_exp_num
        # Launch the xmanager command
        xmanager_cmd = [
            "xmanager", "launch",
            "launch/xm_launch_quadruped_locomotion.py", "--",
            f"--root_dir={root_dir_base.rstrip('/')}/{next_exp_num}",
            f"--participant_module_path={os.path.join('/rl-perf/a2perf/a2perf_benchmark_submission/quadruped_locomotion', algo, debug_path)}",
            f"--motion_file_path={motion_file_path}",
            f"--gin_config={os.path.join(base_gin_config, gin_config_name)}",
            "--local",
            f"--algo={algo}",
            f'--task={motion_file}',
            '--debug' if FLAGS.debug else '',
            f"--seed={seed}",
            f"--mode={FLAGS.mode}",
            f'--skill_level={FLAGS.skill_level}',
            f"--experiment_number={next_exp_num}",
            f'--extra_gin_bindings=Submission.create_domain.motion_files=["{motion_file_path}"]',
            f'--extra_gin_bindings=Submission.create_domain.mode="{env_mode}"',
        ]

        subprocess.run(xmanager_cmd, check=True)


if __name__ == "__main__":
  app.run(main)
