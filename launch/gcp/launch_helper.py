from absl import app
from absl import flags

# Define flags using variables
_MACHINE_NAME = flags.DEFINE_string('machine_name', None,
                                    'Name of the machine to generate commands for')
_ALGO = flags.DEFINE_string('algo', None, 'Algorithm to be used')
_SEED = flags.DEFINE_integer('seed', None, 'Seed for the algorithm')
_NUM_GPUS = flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs to be used')
_DOMAIN = flags.DEFINE_string('domain', None, 'Domain to be used')
_NUM_WEBSITES = flags.DEFINE_integer('num_websites', None,
                                     'Number of websites for web_navigation domain')
_DIFFICULTY_LEVEL = flags.DEFINE_string('difficulty_level', None,
                                        'Difficulty level for web_navigation domain')
_MODE = flags.DEFINE_string('mode', None, 'Mode to be used')
_MOTION_FILE = flags.DEFINE_string('motion_file', None,
                                   'Motion file to be used')
_EXPERIMENT_ID = flags.DEFINE_string('experiment_id', None,
                                     'Experiment ID to be used')
_USER_ID = flags.DEFINE_string('user_id', '1000', 'User ID to be used')
_DEBUG = flags.DEFINE_boolean('debug', False, 'Debug mode')
_NETLISTS = flags.DEFINE_enum('netlists', None,
                              ['toy_macro_stdcell', 'ariane',
                               'macro_tiles_10x10', 'sample_clustered',
                               'simple_grouped_with_coords',
                               'simple_grouped_with_coords_with_blockage',
                               'simple_with_coords', ],
                              'Netlists for the circuit_training domain')
_STD_CELL_PLACER_MODE = flags.DEFINE_enum('std_cell_placer_mode', None,
                                          ['dreamplace', 'plc'],
                                          'Standard cell placer mode for the circuit_training domain')


def generate_commands():
  if _MODE.value == 'inference' and not _EXPERIMENT_ID.value:
    raise ValueError("Experiment ID must be specified for inference mode")

  # Dictionary mapping machine names to addresses
  machine_addresses = {
      "web-nav-0": "10.128.15.228",
      "web-nav-2": "10.128.15.229",
      "web-nav-4": "10.128.15.230",
      "web-nav-6": "10.128.15.231",
      "local": "127.0.0.1"
  }

  address = machine_addresses.get(_MACHINE_NAME.value.lower())
  if address:
    if _DOMAIN.value == 'web_navigation':
      domain_options = (f"--num_websites={_NUM_WEBSITES.value} \\\n"
                        f"--difficulty_levels={_DIFFICULTY_LEVEL.value} \\\n"
                        f"--vocabulary_server_port=50000 \\\n"
                        )
    elif _DOMAIN.value == 'quadruped_locomotion':
      domain_options = f"--motion_files={_MOTION_FILE.value} \\\n"
    elif _DOMAIN.value == 'circuit_training':
      if not _NETLISTS.value or not _STD_CELL_PLACER_MODE.value:
        raise ValueError(
            "Netlists and std_cell_placer_mode must be specified for the circuit_training domain")

      domain_options = f"--netlists={_NETLISTS.value} \\\n--std_cell_placer_mode={_STD_CELL_PLACER_MODE.value} \\\n"
    else:
      raise ValueError(f"Domain {_DOMAIN.value} not supported")

    # Common part of the command
    common_command = (
        f"xmanager launch xm_launch.py -- \\\n"
        f"--domain={_DOMAIN.value} \\\n"
        f"--seeds={_SEED.value} \\\n"
        f"--algos={_ALGO.value} \\\n"
        f"{domain_options}"
        f"--replay_buffer_server_port=8000 \\\n"
        f"--variable_container_server_port=8000 \\\n"
        f"--user_id={_USER_ID.value} \\\n"
        f"--debug={_DEBUG.value} \\\n"
    )

    train_command = common_command + (
        f"--mode=train \\\n"
        f"--num_gpus={_NUM_GPUS.value} \\\n"
        f"--variable_container_server_address=127.0.0.1 \\\n"
        f"--replay_buffer_server_address=127.0.0.1 \\\n"
        f"--vocabulary_server_address=127.0.0.1 \\\n"
        f"--job_type=train "
    )

    inference_command = common_command + (
        f"--mode=inference \\\n"
        f"--num_gpus={_NUM_GPUS.value} \\\n"
        f"--experiment_id={_EXPERIMENT_ID.value} \\\n"
        f"--job_type=inference"
    )

    collect_command = common_command + (
        f"--mode=train \\\n"
        f"--job_type=collect \\\n"
        f"--vocabulary_server_address={address} \\\n"
        f"--replay_buffer_server_address={address} \\\n"
        f"--variable_container_server_address={address} \\\n"
        f"--num_gpus={0} \\\n"
        f"--experiment_id=..."
    )

    if _MODE.value == 'train':
      print(f"Train Command for {_MACHINE_NAME.value}:\n")
      print(train_command)
      print(f"\nCollect Command for {_MACHINE_NAME.value}-collect:\n")
      print(collect_command)
    elif _MODE.value == 'inference':
      print(f"Inference Command for {_MACHINE_NAME.value}:\n")
      print(inference_command)
  else:
    print(f"No configuration found for machine '{_MACHINE_NAME.value}'")


def main(argv):
  del argv  # Unused.
  generate_commands()


if __name__ == "__main__":
  app.run(main)
