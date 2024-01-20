from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('machine_name', None,
                    'Name of the machine to generate commands for')
flags.DEFINE_string('task', None, 'Task to be specified in the commands')


def generate_commands(machine_name, task):
  # Dictionary mapping machine names to addresses
  machine_addresses = {
      "web-nav-0": "10.128.15.228",
      "web-nav-2": "10.128.15.229",
      "web-nav-4": "10.128.15.230",
      "web-nav-6": "10.128.15.231"
  }

  address = machine_addresses.get(machine_name.lower())
  if address:
    train_command = (
        f"xmanager launch launch/xm_launch.py -- \\\n"
        f"--domain=quadruped_locomotion \\\n"
        f"--seeds=4 \\\n"
        f"--algos=ppo \\\n"
        f"--motion_files={task} \\\n"
        f"--mode=train \\\n"
        f"--job_type=train \\\n"
        f"--num_gpus=8 \\\n"
        f"--replay_buffer_server_address=127.0.0.1 \\\n"
        f"--replay_buffer_server_port=8000 \\\n"
        f"--variable_container_server_address=127.0.0.1 \\\n"
        f"--variable_container_server_port=8000"
    )

    collect_command = (
        f"xmanager launch launch/xm_launch.py -- \\\n"
        f"--domain=quadruped_locomotion \\\n"
        f"--seeds=4 \\\n"
        f"--algos=ppo \\\n"
        f"--motion_files={task} \\\n"
        f"--mode=train \\\n"
        f"--job_type=collect \\\n"
        f"--num_gpus=0 \\\n"
        f"--replay_buffer_server_address={address} \\\n"
        f"--replay_buffer_server_port=8000 \\\n"
        f"--variable_container_server_address={address} \\\n"
        f"--variable_container_server_port=8000 \\\n"
        f"--experiment_id=..."
    )

    print(f"Train Command for {machine_name}:\n")
    print(train_command)
    print(f"\nCollect Command for {machine_name}-collect:\n")
    print(collect_command)
  else:
    print(f"No configuration found for machine '{machine_name}'")


def main(argv):
  del argv  # Unused.
  generate_commands(FLAGS.machine_name, FLAGS.task)


if __name__ == "__main__":
  app.run(main)
