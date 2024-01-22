from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('machine_name', None,
                    'Name of the machine to generate commands for')
flags.DEFINE_string('algo', None, 'Algorithm to be used')
flags.DEFINE_integer('seed', None, 'Seed for the algorithm')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs to be used')
flags.DEFINE_string('domain', None, 'Domain to be used')
flags.DEFINE_integer('num_websites', None,
                     'Number of websites for web_navigation domain')
flags.DEFINE_string('difficulty_level', None,
                    'Difficulty level for web_navigation domain')
flags.DEFINE_string('mode', None, 'Mode to be used')
flags.DEFINE_string('motion_file', None, 'Motion file to be used')


def generate_commands(machine_name, algo, seed, domain, num_websites,
    difficulty_level):
  # Dictionary mapping machine names to addresses
  machine_addresses = {
      "web-nav-0": "10.128.15.228",
      "web-nav-2": "10.128.15.229",
      "web-nav-4": "10.128.15.230",
      "web-nav-6": "10.128.15.231",
      "local": "127.0.0.1"
  }

  address = machine_addresses.get(machine_name.lower())
  if address:
    if domain == 'web_navigation':
      domain_options = f"--num_websites={num_websites} \\\n--difficulty_levels={difficulty_level} \\\n"
    elif domain == 'quadruped_locomotion':
      domain_options = f"--motion_files={FLAGS.motion_file} \\\n"
    else:
      raise ValueError(f"Domain {domain} not supported")

    train_command = (
        f"xmanager launch launch/xm_launch.py -- \\\n"
        f"--domain={domain} \\\n"
        f"--seeds={seed} \\\n"
        f"--algos={algo} \\\n"
        f"--mode=train \\\n"
        f"{domain_options}"
        f"--job_type=train \\\n"
        f"--num_gpus={FLAGS.num_gpus} \\\n"
        f"--replay_buffer_server_address=127.0.0.1 \\\n"
        f"--replay_buffer_server_port=8000 \\\n"
        f"--variable_container_server_address=127.0.0.1 \\\n"
        f"--variable_container_server_port=8000"
    )

    collect_command = (
        f"xmanager launch launch/xm_launch.py -- \\\n"
        f"--domain={domain} \\\n"
        f"--seeds={seed} \\\n"
        f"--algos={algo} \\\n"
        f"--mode=train \\\n"
        f"{domain_options}"
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
  generate_commands(FLAGS.machine_name, FLAGS.algo, FLAGS.seed,
                    FLAGS.domain, FLAGS.num_websites, FLAGS.difficulty_level)


if __name__ == "__main__":
  app.run(main)
