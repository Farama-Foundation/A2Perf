from absl import app
from absl import flags
import os
import subprocess

FLAGS = flags.FLAGS

flags.DEFINE_string('motion_file', None, 'Motion file name without extension')
flags.DEFINE_string('algo', None, 'Algorithm to use')
flags.DEFINE_list('seed', None, 'List of seed numbers')
flags.DEFINE_string('experiment_number', None, 'Experiment number')


def get_next_experiment_number(host_dir_base):
  try:
    last_exp_num = \
      sorted([int(d) for d in os.listdir(host_dir_base) if d.isdigit()])[-1]
  except IndexError:
    return "0001"
  return f"{last_exp_num + 1:04d}"


def main(_):
  os.chdir("/home/ikechukwuu/workspace/rl-perf")

  host_dir_base = f"/home/ikechukwuu/workspace/gcs/a2perf/quadruped_locomotion/{FLAGS.motion_file}/{FLAGS.algo}"
  root_dir_base = f"/mnt/gcs/a2perf/quadruped_locomotion/{FLAGS.motion_file}/{FLAGS.algo}"

  seeds = [int(seed) for seed in FLAGS.seed]

  for seed in seeds:
    next_exp_num = get_next_experiment_number(host_dir_base)

    # Docker cleanup
    subprocess.run(["docker", "rm", "-f", "quadruped_locomotion_container"],
                   check=True)

    # Construct the motion file path
    motion_file_path = f"/rl-perf/rl_perf/domains/quadruped_locomotion/motion_imitation/data/motions/{FLAGS.motion_file}.txt"

    # Launch the xmanager command
    xmanager_cmd = [
        "/home/ikechukwuu/workspace/rl-perf/env/bin/xmanager", "launch",
        "launch/xm_launch_quadruped_locomotion.py", "--",
        f"--root_dir={root_dir_base}/{next_exp_num}",
        "--participant_module_path=/rl-perf/rl_perf/rlperf_benchmark_submission/quadruped_locomotion",
        f"--motion_file_path={motion_file_path}",
        "--gin_config=/rl-perf/rl_perf/submission/configs/quadruped_locomotion/train.gin",
        "--local",
        f"--seed={seed}", f"--experiment_number={FLAGS.experiment_number}"
    ]

    subprocess.run(xmanager_cmd, check=True)


if __name__ == "__main__":
  app.run(main)
