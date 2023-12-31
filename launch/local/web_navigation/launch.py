import os
import subprocess

from absl import app
from absl import flags

# Updated to define a list of motion files
_DIFFICULTY_LEVELS = flags.DEFINE_list('difficulty_levels', None,
                                       'List of difficulty levels')
_ALGO = flags.DEFINE_list('algos', None, 'List of algorithms')
_SEED = flags.DEFINE_list('seeds', None, 'List of seed numbers')
_MODE = flags.DEFINE_string('mode', None, 'Mode to run in')
_EXPERIMENT_NUMBER = flags.DEFINE_string(
    'experiment_number', None, 'Experiment number')
_DEBUG = flags.DEFINE_boolean('debug', False, 'Enable debug mode')
_NUM_WEBSITES = flags.DEFINE_list('num_websites', None, 'Number of websites')
_HOST_DIR_BASE = flags.DEFINE_string('host_dir_base',

                                     None, 'Base directory for host')

_SKILL_LEVEL = flags.DEFINE_list('skill_levels', None, 'Skill level')


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

  debug_path = "debug" if _DEBUG.value else ""
  host_dir_base = os.path.expanduser(_HOST_DIR_BASE.value)
  host_dir_base = os.path.abspath(host_dir_base)

  # Docker cleanup, but don't fail if it doesn't exist
  subprocess.run(["docker", "rm", "-f", "web_nav_container"],
                 check=False)

  seeds = [int(seed) for seed in _SEED.value]
  base_gin_config = f'/rl-perf/a2perf/submission/configs/web_navigation/' + debug_path
  gin_config_name = f'train.gin' if _MODE.value == 'train' else f'inference.gin'
  for num_websites in _NUM_WEBSITES.value:
    for algo in _ALGO.value:
      for skill_level in _SKILL_LEVEL.value:
        for difficulty_level in _DIFFICULTY_LEVELS.value:  # Loop through each motion file
          host_dir_base = f'{host_dir_base}/gcs/a2perf/web_navigation/difficulty_level_{difficulty_level}/{algo}/{debug_path}'
          root_dir_base = f"/mnt/gcs/a2perf/web_navigation/difficulty_level_{difficulty_level}/{algo}/{debug_path}"

          for seed in seeds:
            next_exp_num = get_next_experiment_number(host_dir_base)
            next_exp_num = _EXPERIMENT_NUMBER.value if _EXPERIMENT_NUMBER.value is not None else next_exp_num
            print(f"Next experiment number: {next_exp_num}")
            xmanager_cmd = [
                f'env/bin/xmanager', 'launch',
                "launch/xm_launch_web_navigation.py", "--",
                f"--root_dir={root_dir_base.rstrip('/')}/{next_exp_num}",
                f"--participant_module_path={os.path.join('/rl-perf/a2perf/a2perf_benchmark_submission/web_navigation', algo)}",
                f"--difficulty_level={difficulty_level}",
                f"--gin_config={os.path.join(base_gin_config, gin_config_name)}",
                "--local",
                f"--algo={algo}",
                f"--task={difficulty_level}",
                '--debug' if _DEBUG.value else '',
                f"--skill_level={skill_level}",
                f"--seed={seed}",
                f"--num_websites={num_websites}",
                f"--experiment_number={next_exp_num}"]
            try:
              subprocess.run(xmanager_cmd, check=True, stderr=subprocess.PIPE,
                             text=True)
            except subprocess.CalledProcessError as e:
              print("Error occurred:", e.stderr)


if __name__ == "__main__":
  app.run(main)
